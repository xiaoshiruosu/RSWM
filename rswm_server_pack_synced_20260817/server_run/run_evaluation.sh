#!/usr/bin/env bash
# Evaluate one completed experiment through the existing evaluate.py entry.
set -Eeuo pipefail
umask 077

readonly RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly MANIFEST_TOOL="$SCRIPT_DIR/generate_manifest.py"
readonly GPU_SELECTOR="$SCRIPT_DIR/rswm_gpu_selector.py"
readonly PREFLIGHT="$SCRIPT_DIR/preflight_a800.py"

EXPERIMENT_ID=""
EXPERIMENTS_ROOT="${RSWM_EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}"
CLI_GPU=""
DRY_RUN=0
SPLIT_NAME="val"
OUTPUT_DIR=""
CURRENT_STEP="argument_parsing"
PLAN_FILE=""

usage() {
    cat <<'USAGE'
Usage:
  bash server_run/run_evaluation.sh --experiment-id ID [options]

Options:
  --experiments-root PATH   Default: <project>/experiments.
  --gpu PHYSICAL_ID         Default: physical GPU recorded in manifest.yaml.
  --split-name val|test     Default: val. Formal paper rollout explicitly uses test.
  --output-dir PATH         Separate output under the experiment directory. Required
                            for --split-name test so validation artifacts are untouched.
  --dry-run                 Print the existing evaluation command only.
  -h, --help                Show this help.
USAGE
}

while (( $# > 0 )); do
    case "$1" in
        --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
        --experiments-root) EXPERIMENTS_ROOT="${2:-}"; shift 2 ;;
        --gpu) CLI_GPU="${2:-}"; shift 2 ;;
        --split-name) SPLIT_NAME="${2:-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "FATAL: --experiment-id must use only letters, digits, dot, underscore, or dash" >&2
    exit 2
}
case "$SPLIT_NAME" in val|test) ;; *) echo "FATAL: --split-name must be val or test" >&2; exit 2 ;; esac
if [[ "$SPLIT_NAME" == "test" && -z "$OUTPUT_DIR" ]]; then
    echo "FATAL: --split-name test requires a separate --output-dir" >&2
    exit 2
fi
PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly PYTHON
[[ -x "$PYTHON" ]] || { echo "FATAL: Python is not executable: $PYTHON" >&2; exit 2; }

mkdir -p -- "$EXPERIMENTS_ROOT"
EXPERIMENTS_ROOT="$(cd -- "$EXPERIMENTS_ROOT" && pwd -P)"
readonly EXPERIMENTS_ROOT
EXPERIMENT_DIR="$EXPERIMENTS_ROOT/$EXPERIMENT_ID"
CONFIG="$EXPERIMENT_DIR/config.yaml"
[[ -f "$CONFIG" && -f "$EXPERIMENT_DIR/manifest.yaml" ]] || {
    echo "FATAL: experiment config/manifest is incomplete: $EXPERIMENT_DIR" >&2
    exit 2
}
PLAN_FILE="$EXPERIMENTS_ROOT/.evaluation_plan_$$.env"

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
    "$PYTHON" "$MANIFEST_TOOL" status --experiment-dir "$EXPERIMENT_DIR" \
        --status FAILED --message "evaluation step=$CURRENT_STEP line=$line command=$command" \
        --exit-code "$exit_code" >/dev/null 2>&1
    echo "EVALUATION_RESULT=FAIL" >&2
    echo "FAILED_STEP=$CURRENT_STEP line=$line exit_code=$exit_code" >&2
    exit "$exit_code"
}
trap 'on_error $? "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_error 130 "$LINENO" "SIGINT"' INT
trap 'on_error 143 "$LINENO" "SIGTERM"' TERM

CURRENT_STEP="config_plan"
"$PYTHON" "$MANIFEST_TOOL" plan \
    --config "$CONFIG" --project-root "$PROJECT_ROOT" \
    --experiment-id "$EXPERIMENT_ID" --experiments-root "$EXPERIMENTS_ROOT" \
    --output-shell "$PLAN_FILE" >/dev/null
# shellcheck disable=SC1090
source "$PLAN_FILE"

mapfile -d '' -t EVAL_COMMAND < <(
    "$PYTHON" "$MANIFEST_TOOL" command --kind evaluation \
        --config "$CONFIG" --project-root "$PROJECT_ROOT" \
        --experiment-dir "$EXPERIMENT_DIR" --python "$PYTHON" --format nul
)
(( ${#EVAL_COMMAND[@]} > 0 )) || { echo "FATAL: empty evaluation command" >&2; exit 2; }

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$EXPERIMENT_DIR/results"
fi
OUTPUT_DIR="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$OUTPUT_DIR")"
case "$OUTPUT_DIR" in
    "$EXPERIMENT_DIR"/*) ;;
    *) echo "FATAL: --output-dir must stay under $EXPERIMENT_DIR" >&2; exit 2 ;;
esac
OUTPUT_REPLACED=0
for (( i=0; i<${#EVAL_COMMAND[@]}; i++ )); do
    if [[ "${EVAL_COMMAND[$i]}" == "--output_dir" ]]; then
        EVAL_COMMAND[$((i + 1))]="$OUTPUT_DIR"
        OUTPUT_REPLACED=1
        break
    fi
done
(( OUTPUT_REPLACED == 1 )) || { echo "FATAL: evaluation command has no --output_dir" >&2; exit 2; }
[[ "$SPLIT_NAME" == "val" ]] || EVAL_COMMAND+=(--split_name "$SPLIT_NAME")

if [[ -z "$CLI_GPU" ]]; then
    CLI_GPU="$($PYTHON "$MANIFEST_TOOL" field \
        --experiment-dir "$EXPERIMENT_DIR" --name gpu.physical_gpu_id)"
fi

quote_command() {
    local quoted
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

if (( DRY_RUN == 1 )); then
    echo "========== RSWM EVALUATION DRY-RUN =========="
    echo "experiment_id=$EXPERIMENT_ID"
    echo "SEM_ARM_MODE=$PLAN_SEM_ARM_MODE"
    echo "gpu=$CLI_GPU (not accessed)"
    echo "split_name=$SPLIT_NAME"
    echo "lora_checkpoint=$EXPERIMENT_DIR/checkpoints/best_checkpoint"
    echo "results=$OUTPUT_DIR"
    echo "evaluation_command=$(quote_command "${EVAL_COMMAND[@]}")"
    echo "GPU_ACCESSED=0"
    echo "EVALUATION_STARTED=0"
    echo "DRY_RUN_PASS"
    exit 0
fi

[[ -d "$EXPERIMENT_DIR/checkpoints/best_checkpoint" ]] || {
    echo "FATAL: best_checkpoint is missing for evaluation" >&2
    on_error 2 "$LINENO" "verify best_checkpoint"
}
for expert in low_noise high_noise; do
    if [[ ! -f "$EXPERIMENT_DIR/checkpoints/best_checkpoint/$expert/adapter_model.safetensors" \
          && ! -f "$EXPERIMENT_DIR/checkpoints/best_checkpoint/$expert/adapter_model.bin" ]]; then
        echo "FATAL: best_checkpoint is missing $expert adapter" >&2
        on_error 2 "$LINENO" "verify $expert adapter"
    fi
done
[[ "$CLI_GPU" =~ ^[0-9]+$ ]] || {
    echo "FATAL: invalid physical GPU: $CLI_GPU" >&2
    on_error 2 "$LINENO" "validate physical GPU"
}

for dependency in flock nvidia-smi readlink tee; do
    command -v "$dependency" >/dev/null 2>&1 || {
        echo "FATAL: required command is unavailable: $dependency" >&2
        on_error 2 "$LINENO" "locate $dependency"
    }
done

ORIGINAL_CVD="${CUDA_VISIBLE_DEVICES:-}"
GPU_CHECK=("$PYTHON" "$MANIFEST_TOOL" check-gpu-request --gpu "$CLI_GPU")
[[ -n "$ORIGINAL_CVD" ]] && GPU_CHECK+=(--visible-devices "$ORIGINAL_CVD")
"${GPU_CHECK[@]}" >/dev/null

INHERITED_EXPERIMENT_FD="${RSWM_EXPERIMENT_LOCK_FD:-}"
INHERITED_GPU_FD="${RSWM_GPU_LOCK_FD:-}"
if [[ -n "$INHERITED_EXPERIMENT_FD" || -n "$INHERITED_GPU_FD" ]]; then
    [[ "$INHERITED_EXPERIMENT_FD" =~ ^[0-9]+$ && "$INHERITED_GPU_FD" =~ ^[0-9]+$ ]] || {
        echo "FATAL: inherited runner lock descriptors are invalid" >&2
        on_error 2 "$LINENO" "validate inherited lock descriptors"
    }
    EXPECTED_EXPERIMENT_LOCK="$EXPERIMENT_DIR/.runner.lock"
    EXPECTED_GPU_LOCK="$RSWM_BASE/work/rswm_train_gpu_${CLI_GPU}.lock"
    ACTUAL_EXPERIMENT_LOCK="$(readlink -f "/proc/$$/fd/$INHERITED_EXPERIMENT_FD")"
    ACTUAL_GPU_LOCK="$(readlink -f "/proc/$$/fd/$INHERITED_GPU_FD")"
    [[ "$ACTUAL_EXPERIMENT_LOCK" == "$EXPECTED_EXPERIMENT_LOCK" \
       && "$ACTUAL_GPU_LOCK" == "$EXPECTED_GPU_LOCK" ]] || {
        echo "FATAL: inherited lock descriptors do not match this experiment/GPU" >&2
        on_error 2 "$LINENO" "validate inherited lock targets"
    }
    flock -n "$INHERITED_EXPERIMENT_FD" || {
        echo "FATAL: inherited experiment lock is not held" >&2
        on_error 75 "$LINENO" "verify inherited experiment lock"
    }
    flock -n "$INHERITED_GPU_FD" || {
        echo "FATAL: inherited GPU lock is not held" >&2
        on_error 75 "$LINENO" "verify inherited GPU lock"
    }
else
    mkdir -p -- "$RSWM_BASE/work"
    exec 8>"$EXPERIMENT_DIR/.runner.lock"
    flock -n 8 || {
        echo "FATAL: experiment runner already active" >&2
        on_error 75 "$LINENO" "experiment runner lock"
    }
    exec 7>"$RSWM_BASE/work/rswm_train_gpu_${CLI_GPU}.lock"
    flock -n 7 || {
        echo "FATAL: another local RSWM runner owns GPU $CLI_GPU" >&2
        on_error 75 "$LINENO" "physical GPU runner lock"
    }
fi

export CUDA_VISIBLE_DEVICES="$CLI_GPU"
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
export TMP="$TMPDIR" TEMP="$TMPDIR"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export RSWM_PROFILE=official SEM_ARM_MODE="$PLAN_SEM_ARM_MODE"
export RSWM_RUN_ROLE="$PLAN_ROLE"
export DATASET_TIER="$PLAN_DATASET_TIER"
export POSE_MODE="$PLAN_POSE_MODE" RSWM_POSE_MODE="$PLAN_POSE_MODE"
export RSWM_PROMPT_SCHEMA="$PLAN_PROMPT_SCHEMA"
export RSWM_PATH_ESTIMATOR="$PLAN_PATH_ESTIMATOR"
export RSWM_BACKBONE="$PLAN_BACKBONE"
export RSWM_HARDWARE_PROFILE="$PLAN_HARDWARE_PROFILE"
export RSWM_ATTENTION_BACKEND="$PLAN_ATTENTION_BACKEND"
export OFFLOAD="$PLAN_OFFLOAD"

GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"
GPU_MAX_USED_MB="${GPU_MAX_USED_MB:-5000}"
NVIDIA_SMI_BIN="$(command -v nvidia-smi)"
CURRENT_STEP="evaluation_gpu_check"
"$PYTHON" "$GPU_SELECTOR" --check-now "$CLI_GPU" \
    --max-util "$GPU_MAX_UTIL" --max-used-mb "$GPU_MAX_USED_MB" \
    --nvidia-smi "$NVIDIA_SMI_BIN"
CURRENT_STEP="evaluation_preflight"
"$PYTHON" "$PREFLIGHT" --strict --require-single-visible \
    --min-free-gib "$PLAN_MIN_FREE_GIB"

CURRENT_STEP="evaluation"
"$PYTHON" "$MANIFEST_TOOL" status --experiment-dir "$EXPERIMENT_DIR" \
    --status EVALUATING --physical-gpu "$CLI_GPU" \
    --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" \
    --message "command=$(quote_command "${EVAL_COMMAND[@]}")" >/dev/null
echo "[EVALUATION] $(quote_command "${EVAL_COMMAND[@]}")"
EVALUATION_LOG="$EXPERIMENT_DIR/logs/evaluation.log"
[[ "$SPLIT_NAME" == "val" ]] || EVALUATION_LOG="$EXPERIMENT_DIR/logs/evaluation_${SPLIT_NAME}.log"
"${EVAL_COMMAND[@]}" 2>&1 | tee -a "$EVALUATION_LOG"
[[ -f "$OUTPUT_DIR/eval_results.json" ]] || {
    echo "FATAL: evaluation exited 0 but eval_results.json is missing" >&2
    on_error 2 "$LINENO" "verify evaluation result"
}
"$PYTHON" "$MANIFEST_TOOL" status --experiment-dir "$EXPERIMENT_DIR" \
    --status COMPLETED --message "evaluation complete split=$SPLIT_NAME output=$OUTPUT_DIR" >/dev/null
echo "EVALUATION_RESULT=PASS"
echo "split_name=$SPLIT_NAME"
echo "results=$OUTPUT_DIR"
