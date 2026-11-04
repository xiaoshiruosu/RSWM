#!/usr/bin/env bash
# Formal RSWM experiment runner.  It orchestrates the existing train.py only.
set -Eeuo pipefail
umask 077

readonly RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly MANIFEST_TOOL="$SCRIPT_DIR/generate_manifest.py"
readonly GPU_SELECTOR="$SCRIPT_DIR/rswm_gpu_selector.py"
readonly PREFLIGHT="$SCRIPT_DIR/preflight_a800.py"
readonly SCHEDULER_STATUS_FILE="$RSWM_BASE/work/rswm_scheduler_status.env"

CONFIG=""
CLI_GPU=""
EXPERIMENT_ID=""
EXPERIMENTS_ROOT="${RSWM_EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}"
DRY_RUN=0
RESUME=0
RESUME_FROM=""
CURRENT_STEP="argument_parsing"
MANIFEST_READY=0
PLAN_FILE=""
ORIGINAL_CVD_WAS_SET=0
ORIGINAL_CVD=""

usage() {
    cat <<'USAGE'
Usage:
  bash server_run/rswm_train_runner.sh --config CONFIG [options]

Options:
  --config PATH             Required experiment YAML.
  --gpu PHYSICAL_ID         Override hardware.gpu; validated as a physical id.
  --experiment-id ID        Default: YYYYMMDD_<experiment.name>.
  --experiments-root PATH   Default: <project>/experiments.
  --resume                  Resume this experiment from its latest complete checkpoint.
  --resume-from PATH        Explicit complete checkpoint (requires --resume).
  --dry-run                 Print the resolved plan; never checks/uses CUDA or trains.
  -h, --help                Show this help.
USAGE
}

while (( $# > 0 )); do
    case "$1" in
        --config) CONFIG="${2:-}"; shift 2 ;;
        --gpu) CLI_GPU="${2:-}"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
        --experiments-root) EXPERIMENTS_ROOT="${2:-}"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --resume-from) RESUME_FROM="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$CONFIG" ]] || { echo "FATAL: --config is required" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "FATAL: config does not exist: $CONFIG" >&2; exit 2; }
if [[ -n "$RESUME_FROM" && "$RESUME" != "1" ]]; then
    echo "FATAL: --resume-from requires --resume" >&2
    exit 2
fi
if [[ "$RESUME" == "1" && -z "$EXPERIMENT_ID" ]]; then
    echo "FATAL: --resume requires --experiment-id" >&2
    exit 2
fi

if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    ORIGINAL_CVD_WAS_SET=1
    ORIGINAL_CVD="$CUDA_VISIBLE_DEVICES"
fi
readonly ORIGINAL_CVD_WAS_SET ORIGINAL_CVD

PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly PYTHON
[[ -x "$PYTHON" ]] || {
    echo "FATAL: Python is not executable: $PYTHON" >&2
    exit 2
}
[[ -f "$MANIFEST_TOOL" && -f "$GPU_SELECTOR" && -f "$PREFLIGHT" ]] || {
    echo "FATAL: runner support files are incomplete under $SCRIPT_DIR" >&2
    exit 2
}

mkdir -p -- "$EXPERIMENTS_ROOT"
EXPERIMENTS_ROOT="$(cd -- "$EXPERIMENTS_ROOT" && pwd -P)"
readonly EXPERIMENTS_ROOT
PLAN_FILE="$EXPERIMENTS_ROOT/.runner_plan_$$.env"

cleanup() {
    if [[ -n "$PLAN_FILE" && -f "$PLAN_FILE" ]]; then
        rm -f -- "$PLAN_FILE"
    fi
}
trap cleanup EXIT

on_error() {
    local exit_code="$1" line="$2" command="$3"
    trap - ERR INT TERM
    set +e
    if (( MANIFEST_READY == 1 )); then
        "$PYTHON" "$MANIFEST_TOOL" status \
            --experiment-dir "$PLAN_EXPERIMENT_DIR" \
            --status FAILED --message "step=$CURRENT_STEP line=$line command=$command" \
            --exit-code "$exit_code" >/dev/null 2>&1
    fi
    echo "FINAL_RESULT=FAIL" >&2
    echo "FAILED_STEP=$CURRENT_STEP" >&2
    echo "line=$line" >&2
    echo "command=$command" >&2
    echo "exit_code=$exit_code" >&2
    exit "$exit_code"
}
trap 'on_error $? "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_error 130 "$LINENO" "SIGINT"' INT
trap 'on_error 143 "$LINENO" "SIGTERM"' TERM

PLAN_COMMAND=(
    "$PYTHON" "$MANIFEST_TOOL" plan
    --config "$CONFIG" --project-root "$PROJECT_ROOT"
    --experiments-root "$EXPERIMENTS_ROOT" --output-shell "$PLAN_FILE"
)
[[ -n "$EXPERIMENT_ID" ]] && PLAN_COMMAND+=(--experiment-id "$EXPERIMENT_ID")
[[ -n "$CLI_GPU" ]] && PLAN_COMMAND+=(--gpu "$CLI_GPU")
CURRENT_STEP="config_plan"
"${PLAN_COMMAND[@]}" >/dev/null
# Generated atomically from a strict config parser using shlex.quote.
# shellcheck disable=SC1090
source "$PLAN_FILE"

mapfile -d '' -t TRAIN_COMMAND < <(
    "$PYTHON" "$MANIFEST_TOOL" command --kind train \
        --config "$PLAN_CONFIG_PATH" --project-root "$PROJECT_ROOT" \
        --experiment-dir "$PLAN_EXPERIMENT_DIR" --python "$PYTHON" \
        --format nul
)
mapfile -d '' -t EVAL_COMMAND < <(
    "$PYTHON" "$MANIFEST_TOOL" command --kind evaluation \
        --config "$PLAN_CONFIG_PATH" --project-root "$PROJECT_ROOT" \
        --experiment-dir "$PLAN_EXPERIMENT_DIR" --python "$PYTHON" \
        --format nul
)
(( ${#TRAIN_COMMAND[@]} > 0 && ${#EVAL_COMMAND[@]} > 0 )) || {
    echo "FATAL: command planning returned an empty argv" >&2
    exit 2
}

# Resolve resume before dry-run so the displayed argv is the exact command a
# real resume would execute.  This path performs read-only checks in dry-run.
if (( RESUME == 1 )); then
    if [[ -z "$RESUME_FROM" ]]; then
        CURRENT_STEP="find_latest_checkpoint"
        RESUME_FROM="$("$PYTHON" "$MANIFEST_TOOL" latest-checkpoint \
            --experiment-dir "$PLAN_EXPERIMENT_DIR")"
    fi
    CURRENT_STEP="validate_resume_checkpoint"
    RESUME_FROM="$("$PYTHON" "$MANIFEST_TOOL" check-checkpoint \
        --experiment-dir "$PLAN_EXPERIMENT_DIR" --checkpoint "$RESUME_FROM")"
    mapfile -d '' -t TRAIN_COMMAND < <(
        "$PYTHON" "$MANIFEST_TOOL" command --kind train \
            --config "$PLAN_CONFIG_PATH" --project-root "$PROJECT_ROOT" \
            --experiment-dir "$PLAN_EXPERIMENT_DIR" --python "$PYTHON" \
            --resume-from "$RESUME_FROM" --format nul
    )
    (( ${#TRAIN_COMMAND[@]} > 0 )) || {
        echo "FATAL: resume command planning returned an empty argv" >&2
        exit 2
    }
fi

quote_command() {
    local quoted
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

CURRENT_STEP="runtime_path_checks"
"$PYTHON" "$MANIFEST_TOOL" validate \
    --config "$PLAN_CONFIG_PATH" --project-root "$PROJECT_ROOT" \
    --check-paths --experiments-root "$EXPERIMENTS_ROOT"

if (( DRY_RUN == 1 )); then
    echo "================================"
    echo "EXPERIMENT DRY RUN"
    echo "================================"
    echo "experiment_id: $PLAN_EXPERIMENT_ID"
    echo "config: $PLAN_CONFIG_PATH"
    echo "experiment_role: $PLAN_ROLE"
    echo "pose_mode: $PLAN_POSE_MODE"
    echo "SEM_ARM_MODE: $PLAN_SEM_ARM_MODE"
    echo "LoRA rank: $PLAN_LORA_RANK"
    echo "target_modules: $PLAN_LORA_TARGETS"
    echo "model_path: $PLAN_MODEL_CHECKPOINT"
    echo "dataset_path: $PLAN_DATA_TRAIN"
    echo "MODEL_PATH=$PLAN_MODEL_CHECKPOINT"
    echo "DATA_PATH=$PLAN_DATA_TRAIN"
    echo "GPU mode: $PLAN_GPU_REQUEST"
    echo "checkpoint path: $PLAN_EXPERIMENT_DIR/checkpoints"
    echo "PASS: model_path exists"
    echo "PASS: dataset_path exists"
    echo "resume=$RESUME"
    if (( RESUME == 1 )); then
        echo "resume_from=$RESUME_FROM"
    fi
    echo "evaluation_enabled=$PLAN_EVAL_ENABLED"
    echo "train_command=$(quote_command "${TRAIN_COMMAND[@]}")"
    echo "evaluation_command=$(quote_command "${EVAL_COMMAND[@]}")"
    echo "manifest=$PLAN_EXPERIMENT_DIR/manifest.yaml (would be created on a real launch)"
    echo "GPU_ACCESSED=0"
    echo "TRAINING_STARTED=0"
    echo "DRY_RUN_PASS"
    exit 0
fi

for dependency in flock nvidia-smi tee; do
    command -v "$dependency" >/dev/null 2>&1 || {
        echo "FATAL: required command is unavailable: $dependency" >&2
        exit 2
    }
done

export HF_HOME="$RSWM_BASE/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RSWM_BASE/cache/torch"
export XDG_CACHE_HOME="$RSWM_BASE/cache/xdg"
export TRITON_CACHE_DIR="$RSWM_BASE/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RSWM_BASE/cache/torchinductor"
export PIP_CACHE_DIR="$RSWM_BASE/cache/pip"
export CONDA_PKGS_DIRS="$RSWM_BASE/cache/conda"
export TMPDIR="$RSWM_BASE/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p -- "$RSWM_BASE/work" "$RSWM_BASE/logs" "$TMPDIR" \
    "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" \
    "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$PIP_CACHE_DIR" \
    "$CONDA_PKGS_DIRS"

CURRENT_STEP="scheduler_readiness"
[[ -r "$SCHEDULER_STATUS_FILE" ]] || {
    echo "FATAL: scheduler status is missing: $SCHEDULER_STATUS_FILE" >&2
    exit 2
}
STATE=""; SELECTED_GPU=""; DATA_READY=0; CPU_TEST_PASS=0
GPU_PREFLIGHT_PASS=0; DATASET_SMOKE_PASS=0; GPU_SMOKE_PASS=0
# Written atomically by rswm_auto_scheduler.sh with shell-escaped scalars.
# shellcheck disable=SC1090
source "$SCHEDULER_STATUS_FILE"
[[ "$STATE" == "READY_FOR_TRAINING" ]] || {
    echo "FATAL: scheduler state is $STATE, expected READY_FOR_TRAINING" >&2
    exit 2
}
for gate in DATA_READY CPU_TEST_PASS GPU_PREFLIGHT_PASS DATASET_SMOKE_PASS GPU_SMOKE_PASS; do
    [[ "${!gate}" == "1" ]] || {
        echo "FATAL: scheduler readiness gate $gate=${!gate} did not pass" >&2
        exit 2
    }
done
SCHEDULER_GPU="$SELECTED_GPU"

CURRENT_STEP="create_manifest"
MANIFEST_COMMAND=(
    "$PYTHON" "$MANIFEST_TOOL" manifest
    --config "$PLAN_CONFIG_PATH" --project-root "$PROJECT_ROOT"
    --experiment-id "$PLAN_EXPERIMENT_ID" --experiments-root "$EXPERIMENTS_ROOT"
    --python "$PYTHON" --requested-gpu "$PLAN_GPU_REQUEST"
    --status WAITING_GPU
)
if (( RESUME == 1 )); then
    MANIFEST_COMMAND+=(--resume --resume-from "$RESUME_FROM")
fi
"${MANIFEST_COMMAND[@]}" >/dev/null
MANIFEST_READY=1

exec 8>"$PLAN_EXPERIMENT_DIR/.runner.lock"
if ! flock -n 8; then
    echo "FATAL: experiment runner already active: $PLAN_EXPERIMENT_ID" >&2
    on_error 75 "$LINENO" "experiment runner lock"
fi

PHYSICAL_GPU="$PLAN_GPU_REQUEST"
if [[ "$PHYSICAL_GPU" == "auto" ]]; then
    [[ "$SCHEDULER_GPU" =~ ^[0-9]+$ ]] || {
        echo "FATAL: scheduler READY state has no valid selected physical GPU" >&2
        on_error 2 "$LINENO" "resolve scheduler GPU"
    }
    PHYSICAL_GPU="$SCHEDULER_GPU"
fi

CURRENT_STEP="gpu_visibility_constraint"
GPU_REQUEST_COMMAND=(
    "$PYTHON" "$MANIFEST_TOOL" check-gpu-request --gpu "$PHYSICAL_GPU"
)
if (( ORIGINAL_CVD_WAS_SET == 1 )); then
    GPU_REQUEST_COMMAND+=(--visible-devices "$ORIGINAL_CVD")
fi
"${GPU_REQUEST_COMMAND[@]}" >/dev/null

GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-5000}"
NVIDIA_SMI_BIN="$(command -v nvidia-smi)"
CURRENT_STEP="gpu_immediate_check"
"$PYTHON" "$GPU_SELECTOR" --check-now "$PHYSICAL_GPU" \
    --max-util "$GPU_MAX_UTIL" --max-used-mb "$GPU_MAX_USED_MB" \
    --nvidia-smi "$NVIDIA_SMI_BIN"

exec 7>"$RSWM_BASE/work/rswm_train_gpu_${PHYSICAL_GPU}.lock"
if ! flock -n 7; then
    echo "FATAL: another local RSWM runner owns physical GPU $PHYSICAL_GPU" >&2
    on_error 75 "$LINENO" "physical GPU runner lock"
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
PROCESS_LOCAL_DEVICE="cuda:${RSWM_LOCAL_DEVICE_ID:-0}"
echo "PHYSICAL_GPU_ID=$PHYSICAL_GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PROCESS_LOCAL_DEVICE=$PROCESS_LOCAL_DEVICE"
"$PYTHON" "$MANIFEST_TOOL" status \
    --experiment-dir "$PLAN_EXPERIMENT_DIR" --status GPU_SELECTED \
    --physical-gpu "$PHYSICAL_GPU" --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --message "process_local_device=$PROCESS_LOCAL_DEVICE" >/dev/null

CURRENT_STEP="a800_environment_preflight"
"$PYTHON" "$PREFLIGHT" --strict --require-single-visible \
    --min-free-gib "$PLAN_MIN_FREE_GIB"

CURRENT_STEP="immediate_pre_training_gpu_check"
"$PYTHON" "$GPU_SELECTOR" --check-now "$PHYSICAL_GPU" \
    --max-util "$GPU_MAX_UTIL" --max-used-mb "$GPU_MAX_USED_MB" \
    --nvidia-smi "$NVIDIA_SMI_BIN"

export RSWM_PROFILE=official
export SEM_ARM_MODE="$PLAN_SEM_ARM_MODE"
export RSWM_RUN_ROLE="$PLAN_ROLE"
export DATASET_TIER="$PLAN_DATASET_TIER"
export POSE_MODE="$PLAN_POSE_MODE"
export RSWM_POSE_MODE="$PLAN_POSE_MODE"
export RSWM_PROMPT_SCHEMA="$PLAN_PROMPT_SCHEMA"
export RSWM_PATH_ESTIMATOR="$PLAN_PATH_ESTIMATOR"
export RSWM_BACKBONE="$PLAN_BACKBONE"
export RSWM_HARDWARE_PROFILE="$PLAN_HARDWARE_PROFILE"
export RSWM_ATTENTION_BACKEND="$PLAN_ATTENTION_BACKEND"
export OFFLOAD="$PLAN_OFFLOAD"

TRAIN_LOG="$PLAN_EXPERIMENT_DIR/logs/train.log"
CURRENT_STEP="lora_training"
"$PYTHON" "$MANIFEST_TOOL" status \
    --experiment-dir "$PLAN_EXPERIMENT_DIR" --status TRAINING \
    --message "command=$(quote_command "${TRAIN_COMMAND[@]}")" >/dev/null
echo "[TRAIN] $(quote_command "${TRAIN_COMMAND[@]}")"
"${TRAIN_COMMAND[@]}" 2>&1 | tee -a "$TRAIN_LOG"

[[ -d "$PLAN_EXPERIMENT_DIR/checkpoints/best_checkpoint" ]] || {
    echo "FATAL: training exited 0 but best_checkpoint is missing" >&2
    on_error 2 "$LINENO" "verify best_checkpoint"
}
CURRENT_STEP="training_complete"
"$PYTHON" "$MANIFEST_TOOL" status \
    --experiment-dir "$PLAN_EXPERIMENT_DIR" --status TRAINED \
    --message "best_checkpoint=$PLAN_EXPERIMENT_DIR/checkpoints/best_checkpoint" >/dev/null

if [[ "$PLAN_EVAL_ENABLED" == "1" ]]; then
    CURRENT_STEP="evaluation"
    export RSWM_EXPERIMENT_LOCK_FD=8
    export RSWM_GPU_LOCK_FD=7
    bash "$SCRIPT_DIR/run_evaluation.sh" \
        --experiment-id "$PLAN_EXPERIMENT_ID" \
        --experiments-root "$EXPERIMENTS_ROOT" \
        --gpu "$PHYSICAL_GPU"
else
    "$PYTHON" "$MANIFEST_TOOL" status \
        --experiment-dir "$PLAN_EXPERIMENT_DIR" --status COMPLETED \
        --message "training complete; evaluation disabled" >/dev/null
fi

echo "FINAL_RESULT=PASS"
echo "experiment_id=$PLAN_EXPERIMENT_ID"
echo "manifest=$PLAN_EXPERIMENT_DIR/manifest.yaml"
echo "checkpoints=$PLAN_EXPERIMENT_DIR/checkpoints"
echo "results=$PLAN_EXPERIMENT_DIR/results"
