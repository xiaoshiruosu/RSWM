#!/bin/bash
# Capture the exact server environment after preflight; no packages are changed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
OUT="${ENV_CAPTURE_DIR:-$RSWM_BASE/logs/environment/$(date +%Y%m%d_%H%M%S)}"
case "$OUT" in
    "$RSWM_BASE"|"$RSWM_BASE"/*) ;;
    *) echo "FATAL: ENV_CAPTURE_DIR must stay under $RSWM_BASE" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

python "$HERE/preflight_a800.py" --strict --require-single-visible --json > "$OUT/preflight.json"
python -m pip check > "$OUT/pip-check.txt"
python -m pip freeze --all > "$OUT/pip-freeze.txt"
python -VV > "$OUT/python.txt" 2>&1
python -c 'import torch,torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch._C._GLIBCXX_USE_CXX11_ABI)' > "$OUT/torch-build.txt"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
uname -a > "$OUT/uname.txt"

if command -v conda >/dev/null 2>&1; then
    conda list --explicit > "$OUT/conda-explicit.txt"
fi

(cd "$OUT" && for file in *; do
    [ "$file" = "SHA256SUMS" ] || sha256sum -- "$file"
done > SHA256SUMS)
echo "Environment capture PASS: $OUT"
