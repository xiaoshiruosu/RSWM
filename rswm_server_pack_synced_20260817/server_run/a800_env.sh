#!/bin/bash
# Source this file on the shared 4×A800 server. It configures storage/caches
# only; it intentionally never sets CUDA_VISIBLE_DEVICES.
set -euo pipefail

export RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
case "$RSWM_BASE" in
    /nix/rsroll_25828|/nix/rsroll_25828/*) ;;
    *) echo "FATAL: shared-server RSWM_BASE must stay under /nix/rsroll_25828: $RSWM_BASE" >&2; return 2 2>/dev/null || exit 2 ;;
esac

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RSWM_BASE/cache/conda}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RSWM_BASE/cache/pip}"
export HF_HOME="${HF_HOME:-$RSWM_BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$RSWM_BASE/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$RSWM_BASE/cache/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RSWM_BASE/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$RSWM_BASE/cache/torchinductor}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RSWM_BASE/cache/matplotlib}"
export TMPDIR="${TMPDIR:-$RSWM_BASE/tmp}"
export RSWM_TMPDIR="${RSWM_TMPDIR:-$TMPDIR}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$HF_HUB_CACHE" \
    "$HF_DATASETS_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR" "$RSWM_BASE/work" \
    "$RSWM_BASE/logs" "$RSWM_BASE/models"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a800_80g}"
if [ "$HARDWARE_PROFILE" != "a800_80g" ]; then
    echo "FATAL: a800_env.sh requires HARDWARE_PROFILE=a800_80g, got $HARDWARE_PROFILE" >&2
    return 2 2>/dev/null || exit 2
fi
export RSWM_HARDWARE_PROFILE="$HARDWARE_PROFILE"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
case "$ATTENTION_BACKEND" in
    auto|flash_attn|sdpa) ;;
    *) echo "FATAL: invalid ATTENTION_BACKEND=$ATTENTION_BACKEND" >&2; return 2 2>/dev/null || exit 2 ;;
esac
export RSWM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
export OFFLOAD="${OFFLOAD:-auto}"
case "$OFFLOAD" in
    auto|0|1) ;;
    *) echo "FATAL: invalid OFFLOAD=$OFFLOAD (expected auto|0|1)" >&2; return 2 2>/dev/null || exit 2 ;;
esac
# auto is resolved against the currently visible logical GPU by common.sh /
# train.py.  On an 80GB A800 it selects the safe OFFLOAD=1 path.  Explicit 0
# is retained for user-approved throughput/OOM-boundary tests and is never
# selected automatically on A800.
export ACCUM="${ACCUM:-4}"
export MAX_FRAMES="${MAX_FRAMES:-33}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export DEVICE="${DEVICE:-cuda}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "[a800-env] WARNING: CUDA_VISIBLE_DEVICES is unset. On this shared server, select only a GPU allocated to you before smoke/training." >&2
else
    echo "[a800-env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (preserved; not modified)"
fi
echo "[a800-env] RSWM_BASE=$RSWM_BASE TMPDIR=$TMPDIR HF_HOME=$HF_HOME TORCH_HOME=$TORCH_HOME"
echo "[a800-env] profile=$HARDWARE_PROFILE attention=$ATTENTION_BACKEND offload=$OFFLOAD workers=$NUM_WORKERS"
