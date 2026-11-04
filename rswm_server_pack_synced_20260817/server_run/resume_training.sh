#!/usr/bin/env bash
# Resume an existing formal experiment from its newest complete native checkpoint.
set -Eeuo pipefail

readonly RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly MANIFEST_TOOL="$SCRIPT_DIR/generate_manifest.py"
readonly TRAIN_RUNNER="$SCRIPT_DIR/rswm_train_runner.sh"

EXPERIMENT_ID=""
EXPERIMENTS_ROOT="${RSWM_EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}"
CLI_GPU=""
DRY_RUN=0

usage() {
    cat <<'USAGE'
Usage:
  bash server_run/resume_training.sh --experiment-id ID [options]

Options:
  --experiments-root PATH   Default: <project>/experiments.
  --gpu PHYSICAL_ID         Optional physical GPU override.
  --dry-run                 Resolve checkpoint and print the exact resume plan only.
  -h, --help                Show this help.
USAGE
}

while (( $# > 0 )); do
    case "$1" in
        --experiment-id) EXPERIMENT_ID="${2:-}"; shift 2 ;;
        --experiments-root) EXPERIMENTS_ROOT="${2:-}"; shift 2 ;;
        --gpu) CLI_GPU="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "FATAL: --experiment-id must use only letters, digits, dot, underscore, or dash" >&2
    exit 2
}
[[ -d "$EXPERIMENTS_ROOT" ]] || {
    echo "FATAL: experiments root does not exist: $EXPERIMENTS_ROOT" >&2
    exit 2
}
EXPERIMENTS_ROOT="$(cd -- "$EXPERIMENTS_ROOT" && pwd -P)"
readonly EXPERIMENTS_ROOT
EXPERIMENT_DIR="$EXPERIMENTS_ROOT/$EXPERIMENT_ID"
CONFIG="$EXPERIMENT_DIR/config.yaml"
[[ -f "$CONFIG" && -f "$EXPERIMENT_DIR/manifest.yaml" ]] || {
    echo "FATAL: experiment config/manifest is incomplete: $EXPERIMENT_DIR" >&2
    exit 2
}

PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly PYTHON
[[ -x "$PYTHON" ]] || { echo "FATAL: Python is not executable: $PYTHON" >&2; exit 2; }
[[ -f "$MANIFEST_TOOL" && -f "$TRAIN_RUNNER" ]] || {
    echo "FATAL: resume support files are incomplete under $SCRIPT_DIR" >&2
    exit 2
}

LATEST_CHECKPOINT="$("$PYTHON" "$MANIFEST_TOOL" latest-checkpoint \
    --experiment-dir "$EXPERIMENT_DIR")"
echo "experiment_id=$EXPERIMENT_ID"
echo "resume_from=$LATEST_CHECKPOINT"

RUNNER_COMMAND=(
    bash "$TRAIN_RUNNER"
    --config "$CONFIG"
    --experiment-id "$EXPERIMENT_ID"
    --experiments-root "$EXPERIMENTS_ROOT"
    --resume
    --resume-from "$LATEST_CHECKPOINT"
)
[[ -n "$CLI_GPU" ]] && RUNNER_COMMAND+=(--gpu "$CLI_GPU")
(( DRY_RUN == 1 )) && RUNNER_COMMAND+=(--dry-run)

exec "${RUNNER_COMMAND[@]}"
