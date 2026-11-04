#!/usr/bin/env bash
# Shared helpers for the ICLR A800 wrappers. Source only; do not execute directly.
set -Eeuo pipefail

readonly ICLR_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ICLR_PROJECT_ROOT="$(cd -- "$ICLR_SCRIPT_DIR/../.." && pwd -P)"
readonly ICLR_PLAN_TOOL="$ICLR_SCRIPT_DIR/iclr_main_plan.py"
readonly ICLR_PLAN_FILE="$ICLR_PROJECT_ROOT/configs/iclr_main/experiment_plan.json"
readonly ICLR_CONTROL_PYTHON="${RSWM_CONTROL_PYTHON:-python3}"
readonly ICLR_WORKLOAD_PYTHON="${RSWM_PYTHON:-${RSWM_BASE:-/nix/rsroll_25828}/envs/rswm_compat_test/bin/python}"

iclr_die() {
    echo "FATAL: $*" >&2
    exit 2
}

iclr_require_control_python() {
    command -v "$ICLR_CONTROL_PYTHON" >/dev/null 2>&1 || \
        iclr_die "control Python is unavailable: $ICLR_CONTROL_PYTHON"
    [[ -f "$ICLR_PLAN_TOOL" && -f "$ICLR_PLAN_FILE" ]] || \
        iclr_die "ICLR plan files are incomplete under $ICLR_PROJECT_ROOT"
}

iclr_validate_plan() {
    "$ICLR_CONTROL_PYTHON" "$ICLR_PLAN_TOOL" \
        --plan "$ICLR_PLAN_FILE" --rswm-base "$RSWM_BASE" validate "$@"
}

iclr_quote_command() {
    local quoted
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

iclr_print_command() {
    echo "COMMAND=$(iclr_quote_command "$@")"
}

iclr_require_confirmation() {
    local variable="$1" expected="$2" purpose="$3"
    local actual=""
    if [[ -v "$variable" ]]; then
        actual="${!variable}"
    fi
    [[ "$actual" == "$expected" ]] || iclr_die \
        "$purpose requires explicit confirmation: export $variable=$expected"
}

iclr_require_local_asset() {
    local variable="$1" value=""
    if [[ -v "$variable" ]]; then
        value="${!variable}"
    fi
    [[ -n "$value" ]] || iclr_die "required environment variable is unset: $variable"
    [[ -e "$value" ]] || iclr_die "$variable must point to a staged local asset: $value"
}

iclr_require_single_visible_gpu() {
    [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || iclr_die \
        "GPU work requires CUDA_VISIBLE_DEVICES to contain exactly one numeric physical id"
}
