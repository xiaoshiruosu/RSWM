#!/bin/bash
# Verify-first environment setup for the shared A800 server.
# Default behavior is read-only. Installation requires RSWM_ALLOW_INSTALL=1.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$RSWM_BASE/cache/conda}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RSWM_BASE/cache/pip}"
export HF_HOME="${HF_HOME:-$RSWM_BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$RSWM_BASE/cache/torch}"
export TMPDIR="${TMPDIR:-$RSWM_BASE/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$TMPDIR"

EXPECTED_PYTHON="3.10.20"
ALLOW_INSTALL="${RSWM_ALLOW_INSTALL:-0}"
CREATE_CONDA="${CREATE_CONDA:-0}"
ENV_PREFIX="${ENV_PREFIX:-$RSWM_BASE/envs/rswm-a800}"

if [ "$CREATE_CONDA" = "1" ]; then
    [ "$ALLOW_INSTALL" = "1" ] || {
        echo "FATAL: CREATE_CONDA=1 changes the environment; also set RSWM_ALLOW_INSTALL=1 explicitly." >&2
        exit 2
    }
    command -v conda >/dev/null 2>&1 || { echo "FATAL: conda not found" >&2; exit 2; }
    conda create -y -p "$ENV_PREFIX" "python=$EXPECTED_PYTHON"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
fi

if [ "$ALLOW_INSTALL" = "1" ]; then
    echo "Installing the pinned A800 stack without upgrading pip/torch beyond the lock."
    python -m pip install "torch==2.4.0" "torchvision==0.19.0" \
        --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r "$HERE/requirements.txt" \
        -c "$ROOT/requirements-a800-verified.txt"
    if ! python -c "import flash_attn; assert flash_attn.__version__ == '2.7.3'" 2>/dev/null; then
        echo "FATAL: flash-attn 2.7.3 is absent or ABI-incompatible." >&2
        echo "Install the already validated torch2.4/cu121-compatible 2.7.3 wheel; this script will not compile or upgrade it implicitly." >&2
        exit 2
    fi
else
    echo "Verify-only mode (default): no pip/conda changes will be made."
fi

python "$ROOT/server_run/preflight_a800.py" --strict
python -m pip check
echo "Environment verification PASS. For a single-GPU stack smoke, run server_run/run_a800_smoke.sh after selecting CUDA_VISIBLE_DEVICES."
