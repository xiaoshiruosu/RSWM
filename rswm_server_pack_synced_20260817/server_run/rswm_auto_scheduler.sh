#!/usr/bin/env bash
# Unattended, fail-closed readiness scheduler for the shared 4xA800 node.
# It never kills a process, reserves multiple GPUs, or starts formal training.
set -Eeuo pipefail
umask 077

readonly RSWM_BASE="/nix/rsroll_25828"
export RSWM_BASE
export HF_HOME="$RSWM_BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RSWM_BASE/cache/torch"
export CONDA_PKGS_DIRS="$RSWM_BASE/cache/conda"
export PIP_CACHE_DIR="$RSWM_BASE/cache/pip"
export XDG_CACHE_HOME="$RSWM_BASE/cache/xdg"
export TRITON_CACHE_DIR="$RSWM_BASE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RSWM_BASE/cache/torchinductor"
export MPLCONFIGDIR="$RSWM_BASE/cache/matplotlib"
export TMPDIR="$RSWM_BASE/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export RSWM_TMPDIR="$TMPDIR"
# Data readiness must be observable while an external Hugging Face download is
# still allowed to finish.  Offline mode is enabled only after RESISC45 passes
# every readiness gate in wait_for_resisc45().
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HARDWARE_PROFILE=a800_80g
export RSWM_HARDWARE_PROFILE=a800_80g
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
export RSWM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
export OFFLOAD=1
export NUM_WORKERS=0
export DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly SCRIPT_DIR PROJECT_ROOT

readonly WORK_DIR="$RSWM_BASE/work"
readonly LOG_DIR="$RSWM_BASE/logs"
readonly LOCK_FILE="$WORK_DIR/rswm_auto_scheduler.lock"
readonly STATUS_FILE="$WORK_DIR/rswm_scheduler_status.env"
readonly SELECTOR_STATE_FILE="$WORK_DIR/rswm_gpu_selector_status.env"
readonly GPU_RESULT_FILE="$WORK_DIR/rswm_gpu_selection.env"
readonly RUN_ID="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="$LOG_DIR/auto_scheduler_${RUN_ID}.log"
readonly SMOKE_RUN_DIR="$WORK_DIR/rswm_scheduler_smoke_${RUN_ID}"
readonly WHU_SMOKE_H5="$SMOKE_RUN_DIR/whu_smoke.h5"
readonly RESISC_SMOKE_H5="$SMOKE_RUN_DIR/resisc45_smoke.h5"
readonly A800_SMOKE_DIR="$WORK_DIR/a800_smoke"

DATA_POLL_SECONDS="${DATA_POLL_SECONDS:-300}"
GPU_IDLE_CONFIRM_SECONDS="${GPU_IDLE_CONFIRM_SECONDS:-600}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-5000}"
PREFLIGHT_MIN_FREE_GIB="${PREFLIGHT_MIN_FREE_GIB:-70}"
DATASET_SMOKE_SAMPLES="${DATASET_SMOKE_SAMPLES:-12}"
PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly DATA_POLL_SECONDS GPU_IDLE_CONFIRM_SECONDS GPU_POLL_SECONDS
readonly GPU_MAX_UTIL GPU_MAX_USED_MB PREFLIGHT_MIN_FREE_GIB
readonly DATASET_SMOKE_SAMPLES PYTHON

mkdir -p -- "$WORK_DIR" "$LOG_DIR" "$TMPDIR" "$HF_DATASETS_CACHE" \
    "$TORCH_HOME" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR"
exec > >(tee -a "$LOG_FILE") 2>&1

if ! command -v flock >/dev/null 2>&1; then
    echo "FATAL: flock is required for $LOCK_FILE"
    exit 2
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "FATAL: another RSWM scheduler owns $LOCK_FILE; this instance will exit"
    exit 75
fi

STATE="STARTING"
CURRENT_STEP="initialization"
SELECTED_GPU=""
DETAIL=""
CURRENT_COMMAND=""
FAILED_STEP=""
DATA_READY=0
CPU_TEST_PASS=0
GPU_PREFLIGHT_PASS=0
DATASET_SMOKE_PASS=0
GPU_SMOKE_PASS=0

timestamp() {
    date '+%Y-%m-%d %H:%M:%S%z'
}

write_status() {
    local temporary="${STATUS_FILE}.tmp.$$"
    {
        printf 'STATE=%q\n' "$STATE"
        printf 'SELECTED_GPU=%q\n' "$SELECTED_GPU"
        printf 'CURRENT_STEP=%q\n' "$CURRENT_STEP"
        printf 'LAST_UPDATE=%q\n' "$(timestamp)"
        printf 'LOG_FILE=%q\n' "$LOG_FILE"
        printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
        printf 'DETAIL=%q\n' "$DETAIL"
        printf 'DATA_READY=%q\n' "$DATA_READY"
        printf 'CPU_TEST_PASS=%q\n' "$CPU_TEST_PASS"
        printf 'GPU_PREFLIGHT_PASS=%q\n' "$GPU_PREFLIGHT_PASS"
        printf 'DATASET_SMOKE_PASS=%q\n' "$DATASET_SMOKE_PASS"
        printf 'GPU_SMOKE_PASS=%q\n' "$GPU_SMOKE_PASS"
    } >"$temporary"
    mv -f -- "$temporary" "$STATUS_FILE"
}

set_state() {
    STATE="$1"
    CURRENT_STEP="$2"
    DETAIL="${3:-}"
    write_status
    echo "[STATE $(timestamp)] $STATE step=$CURRENT_STEP${DETAIL:+ detail=$DETAIL}"
}

quote_command() {
    local quoted=""
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

on_error() {
    local exit_code="$1"
    local shell_command="${2:-unknown}"
    trap - ERR INT TERM
    set +e
    STATE="FAILED"
    CURRENT_STEP="${FAILED_STEP:-$CURRENT_STEP}"
    DETAIL="exit_code=$exit_code"
    write_status
    echo
    echo "FINAL_RESULT=FAIL"
    echo "FAILED_STEP=$CURRENT_STEP"
    echo "exit_code=$exit_code"
    echo "command=${CURRENT_COMMAND:-$shell_command}"
    echo "shell_command=$shell_command"
    echo "timestamp=$(timestamp)"
    echo "log_file=$LOG_FILE"
    exit "$exit_code"
}
trap 'on_error $? "$BASH_COMMAND"' ERR
trap 'on_error 130 "SIGINT"' INT
trap 'on_error 143 "SIGTERM"' TERM

run_step() {
    local step="$1"
    shift
    local exit_code
    FAILED_STEP="$step"
    CURRENT_STEP="$step"
    CURRENT_COMMAND="$(quote_command "$@")"
    write_status
    echo
    echo "[RUN $(timestamp)] step=$step"
    echo "command=$CURRENT_COMMAND"
    set +e
    "$@"
    exit_code=$?
    set -e
    echo "[EXIT $(timestamp)] step=$step exit_code=$exit_code"
    if (( exit_code != 0 )); then
        return "$exit_code"
    fi
    FAILED_STEP=""
}

require_positive_integer() {
    local name="$1" value="$2"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: $name must be a positive integer, got '$value'"
        return 2
    fi
}

require_nonnegative_integer() {
    local name="$1" value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "FATAL: $name must be a non-negative integer, got '$value'"
        return 2
    fi
}

require_positive_number() {
    local name="$1" value="$2"
    if [[ ! "$value" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
            || [[ "$value" == "0" || "$value" == "0.0" ]]; then
        echo "FATAL: $name must be a positive number, got '$value'"
        return 2
    fi
}

require_file() {
    [[ -f "$1" ]] || { echo "FATAL: required file missing: $1"; return 2; }
}

require_directory() {
    [[ -d "$1" ]] || { echo "FATAL: required directory missing: $1"; return 2; }
}

SNAPSHOT_RESULT=""
resolve_snapshot() {
    local repo_cache="$1" revision snapshot
    SNAPSHOT_RESULT=""
    [[ -f "$repo_cache/refs/main" ]] || return 1
    revision="$(tr -d '\r\n' <"$repo_cache/refs/main")"
    [[ -n "$revision" ]] || return 1
    snapshot="$repo_cache/snapshots/$revision"
    [[ -d "$snapshot" ]] || return 1
    SNAPSHOT_RESULT="$snapshot"
}

incomplete_count() {
    local root="$1"
    if [[ ! -d "$root" ]]; then
        printf '%s' -1
        return
    fi
    find "$root" -type f -name '*.incomplete' -print | wc -l | tr -d '[:space:]'
}

validate_snapshot_cache() {
    local label="$1" repo_cache="$2" count
    require_directory "$repo_cache" || return
    resolve_snapshot "$repo_cache" || {
        echo "FATAL: $label has no refs/main-resolved snapshot: $repo_cache"
        return 2
    }
    count="$(incomplete_count "$repo_cache")"
    [[ "$count" == "0" ]] || {
        echo "FATAL: $label cache still has $count .incomplete file(s)"
        return 2
    }
    echo "[RESOURCE] $label snapshot=$SNAPSHOT_RESULT incomplete=0"
}

check_fixed_resources() {
    local aid_dir whu_dir model_dir clip_dir high_shards low_shards checkpoint
    aid_dir="$HF_HUB_CACHE/datasets--blanchon--AID"
    whu_dir="$HF_HUB_CACHE/datasets--jonathan-roberts1--WHU-RS19"
    model_dir="$HF_HUB_CACHE/models--robbyant--lingbot-world-base-cam"
    clip_dir="$HF_HUB_CACHE/models--openai--clip-vit-base-patch32"

    validate_snapshot_cache AID "$aid_dir" || return
    AID_SNAPSHOT="$SNAPSHOT_RESULT"
    validate_snapshot_cache WHU-RS19 "$whu_dir" || return
    WHU_SNAPSHOT="$SNAPSHOT_RESULT"
    validate_snapshot_cache LingBot-World "$model_dir" || return
    MODEL_SNAPSHOT="$SNAPSHOT_RESULT"
    validate_snapshot_cache CLIP "$clip_dir" || return
    CLIP_SNAPSHOT="$SNAPSHOT_RESULT"

    require_file "$MODEL_SNAPSHOT/Wan2.1_VAE.pth" || return
    require_file "$MODEL_SNAPSHOT/models_t5_umt5-xxl-enc-bf16.pth" || return
    require_file "$MODEL_SNAPSHOT/configuration.json" || return
    require_directory "$MODEL_SNAPSHOT/high_noise_model" || return
    require_directory "$MODEL_SNAPSHOT/low_noise_model" || return
    high_shards="$(find -L "$MODEL_SNAPSHOT/high_noise_model" -maxdepth 1 \
        -type f -name '*.safetensors' -print | wc -l | tr -d '[:space:]')"
    low_shards="$(find -L "$MODEL_SNAPSHOT/low_noise_model" -maxdepth 1 \
        -type f -name '*.safetensors' -print | wc -l | tr -d '[:space:]')"
    if (( high_shards < 8 || low_shards < 8 )); then
        echo "FATAL: LingBot shard count is incomplete: high=$high_shards low=$low_shards"
        return 2
    fi
    echo "[RESOURCE] LingBot shards high=$high_shards low=$low_shards"

    require_file "$CLIP_SNAPSHOT/config.json" || return
    require_file "$CLIP_SNAPSHOT/preprocessor_config.json" || return
    for checkpoint in \
        inception_v3_google-0cc3c7bd.pth \
        r3d_18-b3b3357e.pth \
        alexnet-owt-7be5be79.pth; do
        require_file "$TORCH_HOME/hub/checkpoints/$checkpoint" || return
        echo "[RESOURCE] evaluation weight=$TORCH_HOME/hub/checkpoints/$checkpoint"
    done
}

wait_for_resisc45() {
    local repo_cache="$HF_HUB_CACHE/datasets--blanchon--RESISC45"
    local dir_ok refs_ok snapshot_ok incomplete entries cache_size
    local process_output process_state process_exit ready
    RESISC_SNAPSHOT=""
    while true; do
        dir_ok=0; refs_ok=0; snapshot_ok=0; ready=0
        [[ -d "$repo_cache" ]] && dir_ok=1
        [[ -f "$repo_cache/refs/main" ]] && refs_ok=1
        if resolve_snapshot "$repo_cache"; then
            snapshot_ok=1
            RESISC_SNAPSHOT="$SNAPSHOT_RESULT"
        else
            RESISC_SNAPSHOT=""
        fi
        incomplete="$(incomplete_count "$repo_cache")"
        entries=0
        if [[ -n "$RESISC_SNAPSHOT" ]]; then
            entries="$(find "$RESISC_SNAPSHOT" -mindepth 1 -print | wc -l | tr -d '[:space:]')"
        fi
        cache_size="missing"
        if [[ -d "$repo_cache" ]]; then
            cache_size="$(du -sh "$repo_cache" 2>/dev/null | awk '{print $1}')"
        fi
        process_output=""
        if process_output="$(pgrep -af 'hf.*download.*blanchon/RESISC45' 2>&1)"; then
            process_state="running"
        else
            process_exit=$?
            if (( process_exit == 1 )); then
                process_state="not_running"
                process_output=""
            else
                process_state="check_failed_exit_$process_exit"
            fi
        fi
        if (( dir_ok == 1 && refs_ok == 1 && snapshot_ok == 1 )) \
                && [[ "$incomplete" == "0" && "$process_state" == "not_running" ]]; then
            ready=1
        fi

        set_state "WAITING_DATA" "RESISC45" \
            "entries=$entries incomplete=$incomplete download=$process_state"
        echo "[DATA $(date '+%H:%M:%S')]"
        echo "RESISC45 cache=$repo_cache"
        echo "directory=$dir_ok refs_main=$refs_ok snapshot=$snapshot_ok"
        echo "snapshot_entries=$entries"
        echo "incomplete=$incomplete"
        echo "download_process=$process_state"
        [[ -z "$process_output" ]] || printf '%s\n' "$process_output"
        echo "cache_size=$cache_size"
        if (( ready == 1 )); then
            echo "status=READY"
            break
        fi
        echo "status=WAITING next_poll=${DATA_POLL_SECONDS}s"
        sleep "$DATA_POLL_SECONDS"
    done
    validate_snapshot_cache RESISC45 "$repo_cache"
    RESISC_SNAPSHOT="$SNAPSHOT_RESULT"
}

enable_offline_mode() {
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[MODE $(timestamp)] OFFLINE_MODE_ENABLED"
    echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
    echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"
    echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
}

assert_readiness_gates() {
    local failed=0
    echo "[READY GATES] DATA_READY=$DATA_READY"
    echo "[READY GATES] CPU_TEST_PASS=$CPU_TEST_PASS"
    echo "[READY GATES] GPU_PREFLIGHT_PASS=$GPU_PREFLIGHT_PASS"
    echo "[READY GATES] DATASET_SMOKE_PASS=$DATASET_SMOKE_PASS"
    echo "[READY GATES] GPU_SMOKE_PASS=$GPU_SMOKE_PASS"
    if (( DATA_READY != 1 )); then
        echo "FATAL: READY gate DATA_READY did not pass"
        failed=1
    fi
    if (( CPU_TEST_PASS != 1 )); then
        echo "FATAL: READY gate CPU_TEST_PASS did not pass"
        failed=1
    fi
    if (( GPU_PREFLIGHT_PASS != 1 )); then
        echo "FATAL: READY gate GPU_PREFLIGHT_PASS did not pass"
        failed=1
    fi
    if (( DATASET_SMOKE_PASS != 1 )); then
        echo "FATAL: READY gate DATASET_SMOKE_PASS did not pass"
        failed=1
    fi
    if (( GPU_SMOKE_PASS != 1 )); then
        echo "FATAL: READY gate GPU_SMOKE_PASS did not pass"
        failed=1
    fi
    (( failed == 0 )) || return 2
    echo "[READY GATES] PASS"
}

log_environment() {
    echo "============================================================"
    echo "RSWM unattended A800 readiness scheduler"
    echo "hostname=$(hostname)"
    echo "date=$(timestamp)"
    echo "PROJECT_ROOT=$PROJECT_ROOT"
    echo "RSWM_BASE=$RSWM_BASE"
    echo "LOG_FILE=$LOG_FILE"
    echo "PYTHON=$PYTHON"
    if command -v git >/dev/null 2>&1 \
            && git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "git_root=$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel)"
        echo "git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
    else
        echo "git_commit=NOT_AVAILABLE"
    fi
    echo "CUDA_VISIBLE_DEVICES(original)=${ORIGINAL_CVD_DISPLAY}"
    echo "TMPDIR=$TMPDIR"
    echo "HF_HOME=$HF_HOME"
    echo "HF_HUB_CACHE=$HF_HUB_CACHE"
    echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-<unset>}"
    echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE-<unset>}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE-<unset>}"
    echo "TORCH_HOME=$TORCH_HOME"
    echo "thresholds: util<$GPU_MAX_UTIL used<$GPU_MAX_USED_MB MiB"
    echo "polling: data=${DATA_POLL_SECONDS}s gpu=${GPU_POLL_SECONDS}s confirm=${GPU_IDLE_CONFIRM_SECONDS}s"
    echo "============================================================"
}

restore_original_cuda_visibility() {
    if (( ORIGINAL_CVD_WAS_SET == 1 )); then
        export CUDA_VISIBLE_DEVICES="$ORIGINAL_CVD"
    else
        unset CUDA_VISIBLE_DEVICES
    fi
}

immediate_gpu_check() {
    local label="$1" exit_code
    local command=(
        "$PYTHON" "$SCRIPT_DIR/rswm_gpu_selector.py"
        --check-now "$SELECTED_GPU"
        --baseline-pids "$BASELINE_COMPUTE_PIDS"
        --max-util "$GPU_MAX_UTIL"
        --max-used-mb "$GPU_MAX_USED_MB"
        --nvidia-smi "$NVIDIA_SMI_BIN"
    )
    CURRENT_STEP="$label"
    CURRENT_COMMAND="$(quote_command "${command[@]}")"
    write_status
    echo
    echo "[RUN $(timestamp)] step=$label"
    echo "command=$CURRENT_COMMAND"
    set +e
    "${command[@]}"
    exit_code=$?
    set -e
    echo "[EXIT $(timestamp)] step=$label exit_code=$exit_code"
    return "$exit_code"
}

write_gpu_smoke_marker() {
    local target="$A800_SMOKE_DIR/result.env" temporary="${target}.tmp.$$"
    mkdir -p -- "$A800_SMOKE_DIR"
    {
        printf 'RESULT=PASS\n'
        printf 'PHYSICAL_GPU_ID=%q\n' "$SELECTED_GPU"
        printf 'CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
        printf 'PROCESS_LOCAL_GPU=0\n'
        printf 'COMPLETED_AT=%q\n' "$(timestamp)"
        printf 'LOG_FILE=%q\n' "$LOG_FILE"
    } >"$temporary"
    mv -f -- "$temporary" "$target"
}

main() {
    local dependency selector_exit resource_exit
    local -a selector_command

    ORIGINAL_CVD_WAS_SET=0
    ORIGINAL_CVD=""
    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        ORIGINAL_CVD_WAS_SET=1
        ORIGINAL_CVD="$CUDA_VISIBLE_DEVICES"
        ORIGINAL_CVD_DISPLAY="$CUDA_VISIBLE_DEVICES"
    else
        ORIGINAL_CVD_DISPLAY="<unset>"
    fi
    readonly ORIGINAL_CVD_WAS_SET ORIGINAL_CVD ORIGINAL_CVD_DISPLAY
    cd -- "$PROJECT_ROOT"

    set_state "STARTING" "validate_configuration"
    require_positive_integer DATA_POLL_SECONDS "$DATA_POLL_SECONDS"
    require_positive_integer GPU_IDLE_CONFIRM_SECONDS "$GPU_IDLE_CONFIRM_SECONDS"
    require_positive_integer GPU_POLL_SECONDS "$GPU_POLL_SECONDS"
    require_positive_integer GPU_MAX_UTIL "$GPU_MAX_UTIL"
    require_positive_integer GPU_MAX_USED_MB "$GPU_MAX_USED_MB"
    require_positive_number PREFLIGHT_MIN_FREE_GIB "$PREFLIGHT_MIN_FREE_GIB"
    require_positive_integer DATASET_SMOKE_SAMPLES "$DATASET_SMOKE_SAMPLES"
    if (( DATASET_SMOKE_SAMPLES < 10 || DATASET_SMOKE_SAMPLES % 4 != 0 )); then
        echo "FATAL: DATASET_SMOKE_SAMPLES must be >=10 and divisible by the existing four-scene protocol; got $DATASET_SMOKE_SAMPLES"
        return 2
    fi
    if (( GPU_MAX_UTIL > 100 )); then
        echo "FATAL: GPU_MAX_UTIL must be between 1 and 100, got $GPU_MAX_UTIL"
        return 2
    fi
    case "$ATTENTION_BACKEND" in
        auto|flash_attn|sdpa) ;;
        *) echo "FATAL: ATTENTION_BACKEND=$ATTENTION_BACKEND is invalid"; return 2 ;;
    esac
    for dependency in bash awk du find hostname nvidia-smi pgrep tee tr wc; do
        command -v "$dependency" >/dev/null 2>&1 || {
            echo "FATAL: required command is unavailable: $dependency"
            return 2
        }
    done
    NVIDIA_SMI_BIN="$(command -v nvidia-smi)"
    readonly NVIDIA_SMI_BIN
    [[ -x "$PYTHON" ]] || { echo "FATAL: Python is not executable: $PYTHON"; return 2; }

    require_file "$SCRIPT_DIR/rswm_gpu_selector.py"
    require_file "$SCRIPT_DIR/rswm_scheduler_status.sh"
    require_file "$SCRIPT_DIR/rswm_auto_scheduler_dryrun.sh"
    require_file "$SCRIPT_DIR/preflight_a800.py"
    require_file "$SCRIPT_DIR/a800_smoke.py"
    require_file "$SCRIPT_DIR/check_resisc45.py"
    require_file "$PROJECT_ROOT/lora_remotesense/build_dataset_from_benchmark.py"
    require_file "$PROJECT_ROOT/tests/test_whurs19_data.py"
    require_file "$PROJECT_ROOT/tests/test_resisc45_data.py"
    require_file "$PROJECT_ROOT/tests/test_gpu_scheduler.py"
    mkdir -p -- "$SMOKE_RUN_DIR" "$A800_SMOKE_DIR"

    log_environment
    run_step "Python/PyTorch/CUDA version report" "$PYTHON" -c \
        'import platform, torch; print("python="+platform.python_version()); print("torch="+torch.__version__); print("torch_cuda="+str(torch.version.cuda)); print("cuda_available="+str(torch.cuda.is_available()))'
    run_step "fixed resource inventory" check_fixed_resources
    wait_for_resisc45
    DATA_READY=1
    write_status
    echo "[GATE $(timestamp)] DATA_READY=PASS"

    set_state "CPU_TEST" "static_and_loader_tests"
    enable_offline_mode
    run_step "scheduler shell syntax" bash -n \
        "$SCRIPT_DIR/rswm_auto_scheduler.sh" \
        "$SCRIPT_DIR/rswm_scheduler_status.sh" \
        "$SCRIPT_DIR/rswm_auto_scheduler_dryrun.sh"
    run_step "Python syntax" "$PYTHON" -c \
        'from pathlib import Path; files=[Path("server_run/rswm_gpu_selector.py"),Path("server_run/check_resisc45.py"),Path("lora_remotesense/build_dataset_from_benchmark.py")]; [compile(p.read_text(encoding="utf-8"),str(p),"exec") for p in files]; print("PASS", len(files), "Python files")'
    run_step "WHU-RS19 unit tests" "$PYTHON" -m pytest \
        -p no:cacheprovider tests/test_whurs19_data.py -v
    run_step "RESISC45 unit tests" "$PYTHON" -m pytest \
        -p no:cacheprovider tests/test_resisc45_data.py -v
    run_step "GPU scheduler mock tests" "$PYTHON" -m pytest \
        -p no:cacheprovider tests/test_gpu_scheduler.py -v
    run_step "WHU-RS19 real offline integrity" "$PYTHON" \
        "$SCRIPT_DIR/check_resisc45.py" --tier whurs19 \
        --snapshot-root "$WHU_SNAPSHOT" --check-offline --check-source \
        --report "$SMOKE_RUN_DIR/whurs19_source_check.json"
    run_step "RESISC45 real offline integrity" "$PYTHON" \
        "$SCRIPT_DIR/check_resisc45.py" --tier resisc45 \
        --check-offline --check-source \
        --report "$SMOKE_RUN_DIR/resisc45_source_check.json"
    CPU_TEST_PASS=1
    write_status
    echo "[GATE $(timestamp)] CPU_TEST_PASS=PASS"

    # The official CLI requires four-scene balance, so 12 is the smallest
    # legal smoke size that is not below the requested ten samples.
    run_step "WHU-RS19 real CPU builder smoke" env CUDA_VISIBLE_DEVICES= \
        RSWM_SEMGATE=1 "$PYTHON" \
        "$PROJECT_ROOT/lora_remotesense/build_dataset_from_benchmark.py" \
        --output "$WHU_SMOKE_H5" --instances "$DATASET_SMOKE_SAMPLES" \
        --dataset-tier whurs19 --temp-dir "$SMOKE_RUN_DIR/whu_temp"
    run_step "WHU-RS19 smoke H5 validation" "$PYTHON" \
        "$SCRIPT_DIR/check_resisc45.py" --tier whurs19 \
        --check-h5 "$WHU_SMOKE_H5" --expect-n "$DATASET_SMOKE_SAMPLES" \
        --report "$SMOKE_RUN_DIR/whurs19_h5_check.json"
    run_step "RESISC45 real CPU builder smoke" env CUDA_VISIBLE_DEVICES= \
        RSWM_SEMGATE=1 "$PYTHON" \
        "$PROJECT_ROOT/lora_remotesense/build_dataset_from_benchmark.py" \
        --output "$RESISC_SMOKE_H5" --instances "$DATASET_SMOKE_SAMPLES" \
        --dataset-tier resisc45 --temp-dir "$SMOKE_RUN_DIR/resisc45_temp"
    run_step "RESISC45 smoke H5 validation" "$PYTHON" \
        "$SCRIPT_DIR/check_resisc45.py" --tier resisc45 \
        --check-h5 "$RESISC_SMOKE_H5" --expect-n "$DATASET_SMOKE_SAMPLES" \
        --report "$SMOKE_RUN_DIR/resisc45_h5_check.json"
    DATASET_SMOKE_PASS=1
    write_status
    echo "[GATE $(timestamp)] DATASET_SMOKE_PASS=PASS"

    while true; do
        restore_original_cuda_visibility
        SELECTED_GPU=""
        GPU_PREFLIGHT_PASS=0
        GPU_SMOKE_PASS=0
        set_state "WAITING_GPU" "scan_allowed_physical_gpus"
        selector_command=(
            "$PYTHON" "$SCRIPT_DIR/rswm_gpu_selector.py"
            --confirm-seconds "$GPU_IDLE_CONFIRM_SECONDS"
            --poll-seconds "$GPU_POLL_SECONDS"
            --max-util "$GPU_MAX_UTIL"
            --max-used-mb "$GPU_MAX_USED_MB"
            --result-file "$GPU_RESULT_FILE"
            --state-file "$SELECTOR_STATE_FILE"
            --nvidia-smi "$NVIDIA_SMI_BIN"
        )
        if (( ORIGINAL_CVD_WAS_SET == 1 )); then
            selector_command+=(--visible-devices "$ORIGINAL_CVD")
        fi
        set_state "CONFIRMING_GPU" "continuous_idle_confirmation"
        run_step "stable idle GPU confirmation" "${selector_command[@]}"
        # Created atomically by the trusted helper and mode 0600.
        # shellcheck disable=SC1090
        source "$GPU_RESULT_FILE"
        [[ "${PHYSICAL_GPU_ID:-}" =~ ^[0-9]+$ ]] || {
            echo "FATAL: selector returned invalid PHYSICAL_GPU_ID='${PHYSICAL_GPU_ID:-}'"
            return 2
        }
        SELECTED_GPU="$PHYSICAL_GPU_ID"
        export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
        echo "PHYSICAL_GPU_ID=$SELECTED_GPU"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "PROCESS_LOCAL_GPU=0"

        if immediate_gpu_check "pre-preflight GPU race check"; then
            :
        else
            selector_exit=$?
            if (( selector_exit == 3 )); then
                echo "[GPU] Candidate was claimed before preflight; abandoning it and returning to WAITING_GPU"
                continue
            fi
            on_error "$selector_exit" "$CURRENT_COMMAND"
        fi

        set_state "GPU_PREFLIGHT" "physical_gpu_$SELECTED_GPU"
        if run_step "A800 strict single-GPU preflight" "$PYTHON" \
                "$SCRIPT_DIR/preflight_a800.py" --strict --require-single-visible \
                --min-free-gib "$PREFLIGHT_MIN_FREE_GIB"; then
            :
        else
            selector_exit=$?
            # A resource race is not a software-test failure: abandon the card
            # and wait.  If the card is still idle, preserve the real preflight
            # failure (stack/version/path/etc.) and stop immediately.
            if immediate_gpu_check "post-preflight-failure resource check"; then
                CURRENT_COMMAND="$(quote_command "$PYTHON" \
                    "$SCRIPT_DIR/preflight_a800.py" --strict \
                    --require-single-visible --min-free-gib \
                    "$PREFLIGHT_MIN_FREE_GIB")"
                on_error "$selector_exit" "A800 strict single-GPU preflight"
            else
                resource_exit=$?
                if (( resource_exit == 3 )); then
                    echo "[GPU] Preflight overlapped a new user workload; abandoning GPU and returning to WAITING_GPU"
                    continue
                fi
                on_error "$resource_exit" "$CURRENT_COMMAND"
            fi
        fi
        GPU_PREFLIGHT_PASS=1
        write_status
        echo "[GATE $(timestamp)] GPU_PREFLIGHT_PASS=PASS"

        # Builders already ran CPU-only.  Revalidate their immutable outputs
        # here so the public state machine retains its DATASET_SMOKE gate while
        # no confirmed-idle GPU time is spent on data construction.
        set_state "DATASET_SMOKE" "revalidate_cpu_built_h5"
        run_step "pre-GPU WHU smoke H5 revalidation" "$PYTHON" \
            "$SCRIPT_DIR/check_resisc45.py" --tier whurs19 \
            --check-h5 "$WHU_SMOKE_H5" --expect-n "$DATASET_SMOKE_SAMPLES"
        run_step "pre-GPU RESISC45 smoke H5 revalidation" "$PYTHON" \
            "$SCRIPT_DIR/check_resisc45.py" --tier resisc45 \
            --check-h5 "$RESISC_SMOKE_H5" --expect-n "$DATASET_SMOKE_SAMPLES"

        if immediate_gpu_check "immediate pre-smoke GPU race check"; then
            :
        else
            selector_exit=$?
            if (( selector_exit == 3 )); then
                echo "[GPU] Candidate changed before smoke; abandoning it and returning to WAITING_GPU"
                continue
            fi
            on_error "$selector_exit" "$CURRENT_COMMAND"
        fi

        set_state "GPU_SMOKE" "one_batch_one_optimizer_step"
        run_step "single-A800 LoRA forward/backward smoke" "$PYTHON" \
            "$SCRIPT_DIR/a800_smoke.py" --level 3
        GPU_SMOKE_PASS=1
        write_status
        echo "[GATE $(timestamp)] GPU_SMOKE_PASS=PASS"
        write_gpu_smoke_marker
        run_step "post-smoke physical GPU status" "$NVIDIA_SMI_BIN" \
            --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits
        break
    done

    run_step "final READY_FOR_TRAINING gate assertion" assert_readiness_gates
    set_state "READY_FOR_TRAINING" "all_readiness_gates_passed" \
        "formal_training_not_started"
    echo
    echo "FINAL_RESULT=PASS"
    echo "READY_FOR_TRAINING"
    echo "Formal LoRA training was NOT started."
    echo "WHU_SMOKE_H5=$WHU_SMOKE_H5"
    echo "RESISC_SMOKE_H5=$RESISC_SMOKE_H5"
    echo "A800_SMOKE_DIR=$A800_SMOKE_DIR"
    echo "LOG_FILE=$LOG_FILE"
}

main "$@"
