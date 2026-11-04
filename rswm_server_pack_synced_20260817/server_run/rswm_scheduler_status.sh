#!/usr/bin/env bash
# Read-only status view for rswm_auto_scheduler.sh.  It never signals a process.
set -uo pipefail

readonly RSWM_BASE="/nix/rsroll_25828"
readonly WORK_DIR="$RSWM_BASE/work"
readonly STATUS_FILE="$WORK_DIR/rswm_scheduler_status.env"
readonly SELECTOR_STATE_FILE="$WORK_DIR/rswm_gpu_selector_status.env"
readonly RESISC_DIR="$RSWM_BASE/cache/huggingface/hub/datasets--blanchon--RESISC45"

STATE="NOT_STARTED"
SELECTED_GPU=""
CURRENT_STEP=""
LAST_UPDATE=""
LOG_FILE=""
PROJECT_ROOT=""
DETAIL=""
CANDIDATE_GPU=""
IDLE_ELAPSED_SECONDS=""
SELECTOR_EVENT=""
SELECTOR_LAST_UPDATE=""

echo "RSWM scheduler status — $(date '+%Y-%m-%d %H:%M:%S%z')"
echo "============================================================"

if [[ -r "$STATUS_FILE" ]]; then
    # Written atomically by the scheduler with shell-escaped values.
    # shellcheck disable=SC1090
    source "$STATUS_FILE"
    echo "state=$STATE"
    echo "current_step=$CURRENT_STEP"
    echo "selected_physical_gpu=${SELECTED_GPU:-<none>}"
    echo "last_update=$LAST_UPDATE"
    echo "detail=${DETAIL:-<none>}"
    echo "log_file=${LOG_FILE:-<none>}"
    echo "project_root=${PROJECT_ROOT:-<none>}"
else
    echo "state=NOT_STARTED (status file absent: $STATUS_FILE)"
fi

if [[ -r "$SELECTOR_STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$SELECTOR_STATE_FILE"
    echo "current_candidate_gpu=${CANDIDATE_GPU:-<none>}"
    echo "candidate_idle_seconds=${IDLE_ELAPSED_SECONDS:-0}"
    echo "selector_event=${SELECTOR_EVENT:-unknown}"
    echo "selector_last_update=${SELECTOR_LAST_UPDATE:-unknown}"
else
    echo "current_candidate_gpu=<none>"
fi

echo
echo "Scheduler process (read-only pgrep):"
if pgrep -af '[r]swm_auto_scheduler.sh'; then
    echo "scheduler_running=yes"
else
    echo "scheduler_running=no"
fi

echo
echo "RESISC45 readiness:"
dir_ok=0
refs_ok=0
snapshot_ok=0
snapshot=""
entries=0
incomplete=-1
download_state="not_running"
cache_size="missing"
[[ -d "$RESISC_DIR" ]] && dir_ok=1
if [[ -f "$RESISC_DIR/refs/main" ]]; then
    refs_ok=1
    revision="$(tr -d '\r\n' <"$RESISC_DIR/refs/main")"
    snapshot="$RESISC_DIR/snapshots/$revision"
    [[ -n "$revision" && -d "$snapshot" ]] && snapshot_ok=1
fi
if (( dir_ok == 1 )); then
    incomplete="$(find "$RESISC_DIR" -type f -name '*.incomplete' -print \
        | wc -l | tr -d '[:space:]')"
    cache_size="$(du -sh "$RESISC_DIR" 2>/dev/null | awk '{print $1}')"
fi
if (( snapshot_ok == 1 )); then
    entries="$(find "$snapshot" -mindepth 1 -print | wc -l | tr -d '[:space:]')"
fi
if pgrep -af 'hf.*download.*blanchon/RESISC45' >/dev/null 2>&1; then
    download_state="running"
fi
echo "directory=$dir_ok refs_main=$refs_ok snapshot=$snapshot_ok"
echo "snapshot_entries=$entries incomplete=$incomplete"
echo "download_process=$download_state cache_size=$cache_size"
if (( dir_ok == 1 && refs_ok == 1 && snapshot_ok == 1 )) \
        && [[ "$incomplete" == "0" && "$download_state" == "not_running" ]]; then
    echo "resisc45_status=READY_FOR_CONTENT_VALIDATION"
else
    echo "resisc45_status=WAITING"
fi

echo
echo "Physical GPU status (nvidia-smi; no allocation):"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
else
    echo "nvidia-smi=UNAVAILABLE"
fi
