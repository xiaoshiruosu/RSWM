#!/usr/bin/env bash
# End-to-end, resumable ICLR pipeline for the frozen 2000-sample AID protocol.
set -Eeuo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly PIPELINE_SCHEMA="rswm_iclr_full_pipeline/v1"
readonly DATASET_N=2000

MODE="plan"
WAIT_FOR_DATA=0
POLL_SECONDS=300
GPU="${RSWM_PIPELINE_GPU:-auto}"
EVAL_DATASETS_RAW="${RSWM_PIPELINE_EVAL_DATASETS:-aid}"

usage() {
    cat <<'EOF'
Usage: bash server_run/run_iclr_full_pipeline.sh [options]

Modes (default is read-only):
  --plan                 Print the frozen commands and required assets.
  --status               Report checkpoints/artifacts without changing them.
  --execute              Execute, checkpoint, and resume the full pipeline.

Execution options:
  --wait-for-data        Do not start a build. Poll until the externally running
                         train_geo.h5 becomes a complete, valid 2000-sample file.
  --poll-seconds N       Poll interval for --wait-for-data (default: 300).
  --gpu auto|ID          Training GPU selection; paper evaluation resolves the
                         scheduler-selected GPU when auto (default: auto).
  --eval-datasets aid    This frozen training pipeline evaluates AID only.
                         Cross-dataset runs require their own staged H5/manifests.
  -h, --help             Show this help.

Execution requires all four explicit confirmations:
  RSWM_CONFIRM_FULL_PIPELINE=YES
  RSWM_CONFIRM_FORMAL_RUN=YES
  RSWM_CONFIRM_BASELINE_RUN=YES
  RSWM_CONFIRM_EVALUATION_RUN=YES

Re-running --execute is the resume operation. Matching completed checkpoints
are skipped; a stale/mismatched checkpoint fails closed instead of rerunning.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --plan) MODE="plan"; shift ;;
        --status) MODE="status"; shift ;;
        --execute) MODE="execute"; shift ;;
        --wait-for-data) WAIT_FOR_DATA=1; shift ;;
        --poll-seconds) POLL_SECONDS="${2:-}"; shift 2 ;;
        --gpu) GPU="${2:-}"; shift 2 ;;
        --eval-datasets) EVAL_DATASETS_RAW="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
    echo "FATAL: --poll-seconds must be a positive integer" >&2
    exit 2
}
[[ "$GPU" == "auto" || "$GPU" =~ ^[0-9]+$ ]] || {
    echo "FATAL: --gpu must be auto or a non-negative physical GPU id" >&2
    exit 2
}

[[ "$EVAL_DATASETS_RAW" == "aid" ]] || {
    echo "FATAL: this AID full pipeline accepts only --eval-datasets aid; stage cross-dataset H5/manifests separately" >&2
    exit 2
}
EVAL_DATASETS=(aid)

RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
case "$RSWM_BASE" in
    /nix/rsroll_25828|/nix/rsroll_25828/*) ;;
    *) echo "FATAL: RSWM_BASE must stay under /nix/rsroll_25828: $RSWM_BASE" >&2; exit 2 ;;
esac
export RSWM_BASE

readonly OFFICIAL_ROOT="${RSWM_OFFICIAL_ROOT:-$RSWM_BASE/work/rsroll_official}"
readonly DATASET="$OFFICIAL_ROOT/data/train_geo.h5"
readonly BUILD_TMP="$OFFICIAL_ROOT/tmp_bench_2000"
readonly SPLITS="$OFFICIAL_ROOT/official_splits.json"
readonly MANIFEST_DIR="$OFFICIAL_ROOT/manifests"
readonly AID_MANIFEST="$MANIFEST_DIR/aid_test.json"
readonly REPORT_DIR="$OFFICIAL_ROOT/reports"
readonly STATE_DIR="$OFFICIAL_ROOT/pipeline_state"
readonly LOG_DIR="$OFFICIAL_ROOT/logs/full_pipeline"
readonly EXPERIMENTS_ROOT="${RSWM_EXPERIMENTS_ROOT:-$RSWM_BASE/work/rsroll_iclr_experiments}"
readonly RESULT_ROOT="${RSWM_ICLR_RESULT_ROOT:-$RSWM_BASE/work/rsroll_iclr_baselines}"
readonly TABLE_ROOT="${RSWM_ICLR_TABLE_ROOT:-$RSWM_BASE/work/rsroll_iclr_paper_results}"
readonly PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly CONTROL_PYTHON="${RSWM_CONTROL_PYTHON:-python3}"

export HF_HOME="${HF_HOME:-$RSWM_BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export RSWM_SEMGATE=1

quote_command() {
    local quoted
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

print_plan() {
    local aid_args=()
    [[ -n "${LOCAL_AID_DIR:-}" ]] && aid_args=(--local-aid-dir "$LOCAL_AID_DIR")
    echo "================================"
    echo "ICLR FULL PIPELINE PLAN"
    echo "================================"
    echo "MODE=plan GPU_ACCESSED=0 TRAINING_STARTED=0"
    echo "RSWM_BASE=$RSWM_BASE"
    echo "DATASET=$DATASET N=$DATASET_N"
    echo "SEMANTIC_GATE=required OFFLINE=required"
    echo "EVAL_DATASETS=${EVAL_DATASETS[*]}"
    echo "[01] environment + local model/I3D checks"
    echo "[02] $(quote_command "$PYTHON" "$SCRIPT_DIR/check_aid_pools.py" --threshold 500 "${aid_args[@]}")"
    echo "[03] verify local CLIP snapshot from RSWM_SEMGATE_MODEL"
    if (( WAIT_FOR_DATA == 1 )); then
        echo "[04] wait until $(quote_command "$PYTHON" "$SCRIPT_DIR/check_dataset.py" "$DATASET" --validate --expect-n 2000) passes"
    else
        echo "[04] $(quote_command env N=2000 OUT="$DATASET" TMP="$BUILD_TMP" bash "$SCRIPT_DIR/build_correct.sh")"
    fi
    echo "[05a] $(quote_command "$PYTHON" "$SCRIPT_DIR/check_dataset.py" "$DATASET" --validate --expect-n 2000 --report "$REPORT_DIR/dataset_validation.json")"
    echo "[05b] $(quote_command "$PYTHON" "$SCRIPT_DIR/check_semantic_consistency.py" --gt-h5 "$DATASET" -o "$REPORT_DIR/semantic_consistency.json")"
    echo "[05c] $(quote_command "$PYTHON" "$SCRIPT_DIR/check_benchmark_duplicates.py" --gt-h5 "$DATASET" --expect-zero --expect-n 2000 -o "$REPORT_DIR/duplicate_report.json")"
    echo "[06] $(quote_command "$PYTHON" "$SCRIPT_DIR/make_official_splits.py" --gt-h5 "$DATASET" -o "$SPLITS")"
    echo "[06b] $(quote_command "$PYTHON" "$SCRIPT_DIR/baselines/build_benchmark_manifest.py" --h5 "$DATASET" --split "$SPLITS" --split-name test --source-dataset AID --gen-seed 42 -o "$AID_MANIFEST")"
    local run_id
    for run_id in \
        iclr_primary_native_attn_cam_r16_s42 \
        iclr_primary_native_attn_cam_r16_s43 \
        iclr_primary_native_attn_cam_r16_s44; do
        echo "[07] $(quote_command bash "$SCRIPT_DIR/iclr_a800/run_iclr_main_a800.sh" --execute --group primary --run "$run_id" --experiments-root "$EXPERIMENTS_ROOT")"
    done
    local dataset
    for dataset in "${EVAL_DATASETS[@]}"; do
        local seed alias env_name eval_dir
        for seed in 42 43 44; do
            alias="rsroll_primary_s$seed"
            env_name="RSWM_NATIVE_ATTN_CAM_R16_S${seed}_EVAL_DIR"
            eval_dir="$EXPERIMENTS_ROOT/iclr_primary_native_attn_cam_r16_s${seed}/paper_test/aid"
            echo "[08a/$dataset/$alias] $(quote_command bash "$SCRIPT_DIR/run_evaluation.sh" --experiment-id "iclr_primary_native_attn_cam_r16_s${seed}" --experiments-root "$EXPERIMENTS_ROOT" --split-name test --output-dir "$eval_dir")"
            echo "[08b/$dataset/$alias] $(quote_command env "$env_name=$eval_dir" bash "$SCRIPT_DIR/iclr_a800/run_iclr_baselines_a800.sh" --execute --dataset "$dataset" --group primary --method "$alias" --output-root "$RESULT_ROOT")"
        done
        echo "[08c/$dataset] $(quote_command bash "$SCRIPT_DIR/iclr_a800/run_iclr_evaluation_a800.sh" --execute --evaluation-only --dataset "$dataset" --group primary --metrics paper --result-root "$RESULT_ROOT" --table-root "$TABLE_ROOT")"
        echo "[09/$dataset] $(quote_command bash "$SCRIPT_DIR/iclr_a800/run_iclr_evaluation_a800.sh" --execute --export-only --dataset "$dataset" --group primary --metrics paper --result-root "$RESULT_ROOT" --table-root "$TABLE_ROOT")"
    done
    echo "FINAL_RESULT=PLAN_ONLY"
}

expected_steps() {
    printf '%s\n' 01_environment 02_aid_snapshot 03_clip_snapshot 04_build_dataset \
        05a_dataset_structure 05b_semantic_recheck 05c_duplicate_gate \
        06_official_splits 06b_aid_manifest \
        07_train_s42 07_train_s43 07_train_s44
    local dataset
    for dataset in "${EVAL_DATASETS[@]}"; do
        printf '%s\n' "08a_rollout_${dataset}_s42" "08a_rollout_${dataset}_s43" \
            "08a_rollout_${dataset}_s44" \
            "08b_standardize_${dataset}_s42" "08b_standardize_${dataset}_s43" \
            "08b_standardize_${dataset}_s44" \
            "08c_evaluate_$dataset" "09_tables_$dataset"
    done
}

show_status() {
    echo "PIPELINE_SCHEMA=$PIPELINE_SCHEMA"
    echo "STATE_DIR=$STATE_DIR"
    local step marker state
    while IFS= read -r step; do
        marker="$STATE_DIR/$step.done"
        state="PENDING"
        [[ -f "$marker" ]] && state="DONE"
        printf '%-28s %s\n' "$step" "$state"
    done < <(expected_steps)
    [[ -s "$DATASET" ]] && echo "DATASET_PRESENT=$DATASET" || echo "DATASET_PRESENT=0"
    [[ -s "$SPLITS" ]] && echo "SPLITS_PRESENT=$SPLITS" || echo "SPLITS_PRESENT=0"
    [[ -s "$AID_MANIFEST" ]] && echo "AID_MANIFEST_PRESENT=$AID_MANIFEST" || echo "AID_MANIFEST_PRESENT=0"
    echo "SERVER_STATE_NOT_INFERRED=1"
}

if [[ "$MODE" == "plan" ]]; then
    print_plan
    exit 0
fi
if [[ "$MODE" == "status" ]]; then
    show_status
    exit 0
fi

[[ "${RSWM_CONFIRM_FULL_PIPELINE:-}" == "YES" ]] || {
    echo "FATAL: export RSWM_CONFIRM_FULL_PIPELINE=YES before --execute" >&2
    exit 2
}
[[ "${RSWM_CONFIRM_FORMAL_RUN:-}" == "YES" ]] || {
    echo "FATAL: export RSWM_CONFIRM_FORMAL_RUN=YES before --execute" >&2
    exit 2
}
[[ "${RSWM_CONFIRM_BASELINE_RUN:-}" == "YES" ]] || {
    echo "FATAL: export RSWM_CONFIRM_BASELINE_RUN=YES before --execute" >&2
    exit 2
}
[[ "${RSWM_CONFIRM_EVALUATION_RUN:-}" == "YES" ]] || {
    echo "FATAL: export RSWM_CONFIRM_EVALUATION_RUN=YES before --execute" >&2
    exit 2
}

mkdir -p -- "$STATE_DIR" "$LOG_DIR" "$REPORT_DIR" "$MANIFEST_DIR" \
    "$(dirname -- "$DATASET")"
readonly MASTER_LOG="$LOG_DIR/pipeline.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
exec 9>"$STATE_DIR/pipeline.lock"
flock -n 9 || { echo "FATAL: another full pipeline owns $STATE_DIR/pipeline.lock" >&2; exit 75; }

CURRENT_STEP="bootstrap"
on_exit() {
    local rc="$?"
    if (( rc != 0 )); then
        set +e
        local temporary="$STATE_DIR/.last_failure.$$"
        {
            echo "schema=$PIPELINE_SCHEMA"
            echo "step=$CURRENT_STEP"
            echo "exit_code=$rc"
            echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            echo "master_log=$MASTER_LOG"
        } > "$temporary"
        mv -- "$temporary" "$STATE_DIR/last_failure.env"
        echo "FINAL_RESULT=FAIL step=$CURRENT_STEP exit_code=$rc" >&2
    fi
}
trap on_exit EXIT

hash_text() {
    local output
    output="$(printf '%s' "$1" | sha256sum)"
    printf '%s' "${output%% *}"
}

file_sha256() {
    local output
    output="$(sha256sum -- "$1")"
    printf '%s' "${output%% *}"
}

file_identity() {
    [[ -f "$1" ]] || return 1
    stat -Lc '%s:%Y' -- "$1"
}

write_checkpoint() {
    local step="$1" signature="$2" disposition="${3:-executed}"
    local temporary
    temporary="$(mktemp "$STATE_DIR/.${step}.XXXXXX")"
    {
        echo "schema=$PIPELINE_SCHEMA"
        echo "step=$step"
        echo "signature=$signature"
        echo "disposition=$disposition"
        echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$temporary"
    mv -- "$temporary" "$STATE_DIR/$step.done"
}

run_step() {
    local step="$1" description="$2" signature="$3"
    local verify_function="$4" action_function="$5" adopt="${6:-0}"
    local marker="$STATE_DIR/$step.done" step_log="$LOG_DIR/$step.log"
    CURRENT_STEP="$step"
    if [[ -f "$marker" ]]; then
        if grep -Fqx "schema=$PIPELINE_SCHEMA" "$marker" && \
           grep -Fqx "signature=$signature" "$marker" && \
           "$verify_function" >/dev/null 2>&1; then
            echo "[skip] $step $description (checkpoint and artifacts verified)"
            return 0
        fi
        echo "FATAL: stale or unverifiable checkpoint: $marker" >&2
        echo "       inspect its step log and artifacts; remove only this marker after an explicit decision" >&2
        return 20
    fi
    if [[ "$adopt" == "1" ]] && "$verify_function" >/dev/null 2>&1; then
        write_checkpoint "$step" "$signature" adopted
        echo "[adopt] $step $description (pre-existing complete artifacts)"
        return 0
    fi
    {
        echo ""
        echo "===== START $step: $description ====="
        echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } | tee -a "$step_log"
    if "$action_function" 2>&1 | tee -a "$step_log"; then
        if ! "$verify_function" >/dev/null 2>&1; then
            echo "FATAL: $step command exited zero but artifact verification failed" | tee -a "$step_log" >&2
            return 21
        fi
        write_checkpoint "$step" "$signature" executed
        echo "===== PASS $step =====" | tee -a "$step_log"
    else
        local rc="${PIPESTATUS[0]}"
        echo "===== FAIL $step exit_code=$rc =====" | tee -a "$step_log" >&2
        return "$rc"
    fi
}

verify_noop() { return 0; }

check_environment() {
    local command
    for command in bash flock grep mktemp mv sha256sum stat tee nvidia-smi; do
        command -v "$command" >/dev/null 2>&1 || { echo "missing command: $command" >&2; return 2; }
    done
    [[ -x "$PYTHON" ]] || { echo "Python is not executable: $PYTHON" >&2; return 2; }
    command -v "$CONTROL_PYTHON" >/dev/null 2>&1 || { echo "control Python missing: $CONTROL_PYTHON" >&2; return 2; }
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/a800_env.sh"
    "$PYTHON" -c 'import h5py, numpy, PIL, torch, transformers; print("python_dependencies=PASS")'
    "$CONTROL_PYTHON" "$SCRIPT_DIR/iclr_a800/iclr_main_plan.py" \
        --rswm-base "$RSWM_BASE" validate
    [[ -d "$RSWM_BASE/models/lingbot-world-base-cam" ]] || {
        echo "missing local base model: $RSWM_BASE/models/lingbot-world-base-cam" >&2; return 2; }
    [[ -n "${RSWM_I3D_CHECKPOINT:-}" && -f "$RSWM_I3D_CHECKPOINT" ]] || {
        echo "RSWM_I3D_CHECKPOINT must name a staged local file" >&2; return 2; }
    nvidia-smi -L
    if [[ "$GPU" =~ ^[0-9]+$ ]]; then
        nvidia-smi -i "$GPU" --query-gpu=index,name,memory.total --format=csv,noheader
    fi
    df -h -- "$RSWM_BASE"
}

aid_snapshot_present() {
    if [[ -n "${LOCAL_AID_DIR:-}" ]]; then
        [[ "$LOCAL_AID_DIR" == /* && -d "$LOCAL_AID_DIR" ]]
        return
    fi
    local root="$HF_HUB_CACHE/datasets--blanchon--AID/snapshots"
    [[ -d "$root" ]] || return 1
    local snapshots=("$root"/*)
    [[ -d "${snapshots[0]:-}" ]]
}

check_aid_snapshot() {
    aid_snapshot_present || {
        echo "no local AID source: set LOCAL_AID_DIR or stage blanchon/AID under $HF_HUB_CACHE" >&2
        return 2
    }
    local command=("$PYTHON" "$SCRIPT_DIR/check_aid_pools.py" --threshold 500)
    [[ -n "${LOCAL_AID_DIR:-}" ]] && command+=(--local-aid-dir "$LOCAL_AID_DIR")
    "${command[@]}"
}

clip_snapshot_present() {
    [[ "${RSWM_SEMGATE:-}" == "1" ]]
    [[ -n "${RSWM_SEMGATE_MODEL:-}" && "$RSWM_SEMGATE_MODEL" == /* ]]
    [[ -d "$RSWM_SEMGATE_MODEL" && -r "$RSWM_SEMGATE_MODEL/config.json" ]]
    [[ -r "$RSWM_SEMGATE_MODEL/preprocessor_config.json" ]]
    [[ -r "$RSWM_SEMGATE_MODEL/model.safetensors" || \
       -r "$RSWM_SEMGATE_MODEL/pytorch_model.bin" ]]
}

check_clip_snapshot() {
    clip_snapshot_present || {
        echo "RSWM_SEMGATE_MODEL is not a complete local CLIP snapshot" >&2
        return 2
    }
    "$PYTHON" -c 'import os; from transformers import CLIPModel, CLIPProcessor; p=os.environ["RSWM_SEMGATE_MODEL"]; CLIPProcessor.from_pretrained(p, local_files_only=True); CLIPModel.from_pretrained(p, local_files_only=True); print("local_clip_snapshot=PASS")'
}

verify_dataset() {
    [[ -s "$DATASET" ]] || return 1
    "$PYTHON" "$SCRIPT_DIR/check_dataset.py" "$DATASET" \
        --validate --expect-n "$DATASET_N" >/dev/null 2>&1
}

wait_or_build_dataset() {
    if (( WAIT_FOR_DATA == 1 )); then
        echo "[wait] external build mode; no build command will be launched"
        while true; do
            if [[ -s "$DATASET" ]] && \
               "$PYTHON" "$SCRIPT_DIR/check_dataset.py" "$DATASET" \
                   --validate --expect-n "$DATASET_N" >/dev/null 2>&1; then
                echo "[wait] complete canonical dataset detected: $DATASET"
                return 0
            fi
            echo "[wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) dataset not complete; next check in ${POLL_SECONDS}s"
            sleep "$POLL_SECONDS"
        done
    fi
    N="$DATASET_N" OUT="$DATASET" TMP="$BUILD_TMP" PY="$PYTHON" \
        bash "$SCRIPT_DIR/build_correct.sh"
}

verify_structure_report() {
    [[ -s "$REPORT_DIR/dataset_validation.json" ]] || return 1
    "$PYTHON" - "$REPORT_DIR/dataset_validation.json" "$DATASET" <<'PY'
import json, os, sys
r=json.load(open(sys.argv[1], encoding="utf-8")); s=os.stat(sys.argv[2])
assert r["schema"] == "rswm_dataset_validation/v1" and r["status"] == "PASS"
assert r["summary"]["n_samples"] == 2000
assert r["input"]["bytes"] == s.st_size and r["input"]["mtime_ns"] == s.st_mtime_ns
PY
}

validate_structure() {
    "$PYTHON" "$SCRIPT_DIR/check_dataset.py" "$DATASET" --validate \
        --expect-n "$DATASET_N" --report "$REPORT_DIR/dataset_validation.json"
}

verify_semantic_report() {
    [[ -s "$REPORT_DIR/semantic_consistency.json" ]] || return 1
    "$PYTHON" - "$REPORT_DIR/semantic_consistency.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["n_semantic"] > 0 and r["n_fail"] == 0
assert not r["bad_attr"] and not r["enabled_mismatch"]
assert r["scope"]["n_examined"] == r["n_semantic"]
PY
}

validate_semantic() {
    "$PYTHON" "$SCRIPT_DIR/check_semantic_consistency.py" --gt-h5 "$DATASET" \
        -o "$REPORT_DIR/semantic_consistency.json"
}

verify_duplicate_report() {
    [[ -s "$REPORT_DIR/duplicate_report.json" ]] || return 1
    "$PYTHON" - "$REPORT_DIR/duplicate_report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], encoding="utf-8")); s=r["source_identity"]
assert r["n_samples"] == 2000
assert not r["trajectory_template_dups"] and not r["video_content_dups"]
assert not r["scene_image_dups"] and not s["duplicate_source_groups"]
assert s["n_missing_attrs"] == 0 and s["n_missing_path"] == 0
PY
}

validate_duplicates() {
    "$PYTHON" "$SCRIPT_DIR/check_benchmark_duplicates.py" --gt-h5 "$DATASET" \
        --expect-zero --expect-n "$DATASET_N" -o "$REPORT_DIR/duplicate_report.json"
}

verify_splits() {
    [[ -s "$SPLITS" ]] || return 1
    "$PYTHON" - "$SPLITS" "$DATASET" <<'PY'
import json, os, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["schema"] == "rswm_official_splits/v1" and r["self_check"]["pass"]
assert r["dataset"]["n_samples"] == 2000
assert os.path.realpath(r["dataset"]["path"]) == os.path.realpath(sys.argv[2])
assert sum(r["splits"][x]["n"] for x in ("train","val","test")) == 2000
PY
}

make_splits() {
    "$PYTHON" "$SCRIPT_DIR/make_official_splits.py" --gt-h5 "$DATASET" -o "$SPLITS"
}

verify_aid_manifest() {
    [[ -s "$AID_MANIFEST" ]] || return 1
    "$PYTHON" - "$AID_MANIFEST" "$DATASET" "$SPLITS" <<'PY'
import json, os, sys
r=json.load(open(sys.argv[1], encoding="utf-8"))
assert r["schema"] == "rsroll.benchmark_manifest/v2"
assert r["headline_eligible"] is True and r["source_dataset"] == "AID"
assert r["source_split"] == "test" and r["gen_seed"] == 42
assert r["frame_counts"] == [17, 25, 33] and len(r["sample_keys"]) == 400
assert os.path.realpath(r["h5"]) == os.path.realpath(sys.argv[2])
assert os.path.realpath(r["split"]) == os.path.realpath(sys.argv[3])
PY
}

build_aid_manifest() {
    "$PYTHON" "$SCRIPT_DIR/baselines/build_benchmark_manifest.py" \
        --h5 "$DATASET" --split "$SPLITS" --split-name test \
        --source-dataset AID --gen-seed 42 -o "$AID_MANIFEST"
}

ACTIVE_RUN=""
train_artifacts_ok() {
    local root="$EXPERIMENTS_ROOT/$ACTIVE_RUN"
    [[ -f "$root/manifest.yaml" && -d "$root/checkpoints/best_checkpoint" \
       && -s "$root/results/eval_results.json" ]] || return 1
    grep -Eq '^status:[[:space:]]+COMPLETED[[:space:]]*$' "$root/manifest.yaml"
}

train_active_run() {
    local command=(bash "$SCRIPT_DIR/iclr_a800/run_iclr_main_a800.sh"
        --execute --group primary --run "$ACTIVE_RUN"
        --experiments-root "$EXPERIMENTS_ROOT")
    [[ "$GPU" =~ ^[0-9]+$ ]] && command+=(--gpu "$GPU")
    local experiment_dir="$EXPERIMENTS_ROOT/$ACTIVE_RUN"
    if [[ -f "$experiment_dir/manifest.yaml" ]] && ! train_artifacts_ok && \
       "$PYTHON" "$SCRIPT_DIR/generate_manifest.py" latest-checkpoint \
           --experiment-dir "$experiment_dir" >/dev/null 2>&1; then
        command+=(--resume)
        echo "[resume] $ACTIVE_RUN from its latest validated native checkpoint"
    fi
    "${command[@]}"
}

paper_rollout_artifacts_ok() {
    local root="$EXPERIMENTS_ROOT/$ACTIVE_RUN/paper_test/aid"
    [[ -s "$root/eval_results.json" && -s "$root/run_manifest.json" ]] || return 1
    "$PYTHON" - "$root" "$AID_MANIFEST" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); manifest=json.load(open(sys.argv[2], encoding="utf-8"))
run=json.load(open(root / "run_manifest.json", encoding="utf-8"))
assert run.get("split_name") == "test"
keys=[str(x) for x in manifest["sample_keys"]]
assert len(keys) == 400
for key in keys:
    assert (root / "generated" / f"{key}.npy").is_file()
    assert (root / "gt" / f"{key}.npy").is_file()
    assert (root / "meta" / f"{key}.json").is_file()
PY
}

generate_primary_test_rollout() {
    local output="$EXPERIMENTS_ROOT/$ACTIVE_RUN/paper_test/aid"
    local command=(bash "$SCRIPT_DIR/run_evaluation.sh"
        --experiment-id "$ACTIVE_RUN" --experiments-root "$EXPERIMENTS_ROOT"
        --split-name test --output-dir "$output")
    [[ "$GPU" =~ ^[0-9]+$ ]] && command+=(--gpu "$GPU")
    "${command[@]}"
}

resolve_evaluation_gpu() {
    if [[ "$GPU" =~ ^[0-9]+$ ]]; then
        printf '%s' "$GPU"
        return
    fi
    if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]]; then
        printf '%s' "$CUDA_VISIBLE_DEVICES"
        return
    fi
    local status_file="$RSWM_BASE/work/rswm_scheduler_status.env"
    [[ -r "$status_file" ]] || return 1
    local selected
    selected="$(bash -c 'set -euo pipefail; source "$1"; [[ "${STATE:-}" == READY_FOR_TRAINING ]]; printf "%s" "${SELECTED_GPU:-}"' _ "$status_file")"
    [[ "$selected" =~ ^[0-9]+$ ]] || return 1
    printf '%s' "$selected"
}

ACTIVE_DATASET=""
ACTIVE_SEED=""
baseline_rows() {
    "$CONTROL_PYTHON" "$SCRIPT_DIR/iclr_a800/iclr_main_plan.py" \
        --rswm-base "$RSWM_BASE" list-baselines --group primary --format tsv
}

standardized_artifacts_ok() {
    local root="$RESULT_ROOT/$ACTIVE_DATASET/rsroll_primary_s$ACTIVE_SEED"
    [[ -s "$root/aggregate_metrics.json" && -s "$root/frames.npz" \
       && -s "$root/sample_metrics.jsonl" && -s "$root/manifest.json" \
       && -s "$root/config.json" && -s "$root/baseline_card.json" ]]
}

standardize_active_seed() {
    local alias="rsroll_primary_s$ACTIVE_SEED"
    local variable="RSWM_NATIVE_ATTN_CAM_R16_S${ACTIVE_SEED}_EVAL_DIR"
    local source="$EXPERIMENTS_ROOT/iclr_primary_native_attn_cam_r16_s${ACTIVE_SEED}/paper_test/aid"
    [[ -d "$source/generated" && -d "$source/gt" && -d "$source/meta" ]] || {
        echo "formal evaluation output is incomplete: $source" >&2
        return 2
    }
    export "$variable=$source"
    bash "$SCRIPT_DIR/iclr_a800/run_iclr_baselines_a800.sh" --execute \
        --dataset "$ACTIVE_DATASET" --group primary --method "$alias" \
        --output-root "$RESULT_ROOT"
}

evaluation_artifacts_ok() {
    local rows alias _rest
    rows="$(baseline_rows)" || return 1
    while IFS=$'\t' read -r alias _rest; do
        [[ -n "$alias" ]] || continue
        [[ -s "$RESULT_ROOT/$ACTIVE_DATASET/$alias/aggregate_metrics.json" \
           && -s "$RESULT_ROOT/$ACTIVE_DATASET/$alias/frames.npz" ]] || return 1
    done <<< "$rows"
}

evaluate_active_dataset() {
    local eval_gpu
    eval_gpu="$(resolve_evaluation_gpu)" || {
        echo "cannot resolve one scheduler-approved GPU for paper evaluation" >&2
        return 2
    }
    export CUDA_VISIBLE_DEVICES="$eval_gpu"
    bash "$SCRIPT_DIR/iclr_a800/run_iclr_evaluation_a800.sh" --execute \
        --evaluation-only --dataset "$ACTIVE_DATASET" --group primary --metrics paper \
        --result-root "$RESULT_ROOT" --table-root "$TABLE_ROOT"
}

table_artifacts_ok() {
    local root="$TABLE_ROOT/$ACTIVE_DATASET" file
    [[ -s "$root/export_manifest.json" ]] || return 1
    for file in main_results.csv overlap_results.csv trajectory_results.csv \
        rank_ablation.csv target_ablation.csv cross_dataset_results.csv \
        efficiency_results.csv; do
        [[ -s "$root/$file" ]] || return 1
    done
}

export_active_tables() {
    bash "$SCRIPT_DIR/iclr_a800/run_iclr_evaluation_a800.sh" --execute \
        --export-only --dataset "$ACTIVE_DATASET" --group primary --metrics paper \
        --result-root "$RESULT_ROOT" --table-root "$TABLE_ROOT"
}

echo "PIPELINE_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "MASTER_LOG=$MASTER_LOG"
echo "SERVER_STATE_CHECKED_BY_SCRIPT=1"

env_signature="$(hash_text "$PIPELINE_SCHEMA|env|$PYTHON|$CONTROL_PYTHON|$GPU|${RSWM_I3D_CHECKPOINT:-unset}")"
run_step 01_environment "check local environment and observed GPU inventory" \
    "$env_signature" verify_noop check_environment

aid_signature="$(hash_text "$PIPELINE_SCHEMA|aid|${LOCAL_AID_DIR:-$HF_HUB_CACHE/datasets--blanchon--AID}|threshold=500")"
run_step 02_aid_snapshot "check offline AID snapshot and disjoint source pools" \
    "$aid_signature" aid_snapshot_present check_aid_snapshot

clip_signature="$(hash_text "$PIPELINE_SCHEMA|clip|${RSWM_SEMGATE_MODEL:-unset}")"
run_step 03_clip_snapshot "check offline local CLIP snapshot" \
    "$clip_signature" clip_snapshot_present check_clip_snapshot

build_signature="$(hash_text "$PIPELINE_SCHEMA|build|N=2000|AID|128x222|480x832|steps9|frames17,25,33|overlap0,0.1|semgate1|wait=$WAIT_FOR_DATA")"
run_step 04_build_dataset "obtain canonical train_geo.h5" \
    "$build_signature" verify_dataset wait_or_build_dataset 1

dataset_identity="$(file_identity "$DATASET")"
structure_signature="$(hash_text "$PIPELINE_SCHEMA|structure|$dataset_identity")"
run_step 05a_dataset_structure "validate HDF5 structure and frozen protocol" \
    "$structure_signature" verify_structure_report validate_structure

semantic_signature="$(hash_text "$PIPELINE_SCHEMA|semantic|$dataset_identity|${RSWM_SEMGATE_MODEL}")"
run_step 05b_semantic_recheck "independently re-run the offline SemanticGate" \
    "$semantic_signature" verify_semantic_report validate_semantic

duplicate_signature="$(hash_text "$PIPELINE_SCHEMA|duplicates|$dataset_identity|N=2000")"
run_step 05c_duplicate_gate "reject duplicate/source-reused samples" \
    "$duplicate_signature" verify_duplicate_report validate_duplicates

split_signature="$(hash_text "$PIPELINE_SCHEMA|split|$dataset_identity|seed42|70,10,20|source-disjoint")"
run_step 06_official_splits "make source-disjoint official split" \
    "$split_signature" verify_splits make_splits 1

split_hash="$(file_sha256 "$SPLITS")"
manifest_signature="$(hash_text "$PIPELINE_SCHEMA|aid-manifest|$dataset_identity|$split_hash|test|seed42")"
run_step 06b_aid_manifest "freeze the 400-sample AID test manifest" \
    "$manifest_signature" verify_aid_manifest build_aid_manifest 1

aid_manifest_hash="$(file_sha256 "$AID_MANIFEST")"
for seed in 42 43 44; do
    ACTIVE_RUN="iclr_primary_native_attn_cam_r16_s${seed}"
    config="$PROJECT_ROOT/configs/iclr_main/primary_native_attn_cam_r16_s${seed}.yaml"
    train_signature="$(hash_text "$PIPELINE_SCHEMA|train|$ACTIVE_RUN|$dataset_identity|$split_hash|$aid_manifest_hash|$(file_sha256 "$config")")"
    run_step "07_train_s${seed}" "train/evaluate primary LoRA seed $seed through rswm_train_runner" \
        "$train_signature" train_artifacts_ok train_active_run 1
done

training_state_hash="$(hash_text "$(file_sha256 "$STATE_DIR/07_train_s42.done")|$(file_sha256 "$STATE_DIR/07_train_s43.done")|$(file_sha256 "$STATE_DIR/07_train_s44.done")")"
for ACTIVE_DATASET in "${EVAL_DATASETS[@]}"; do
    for ACTIVE_SEED in 42 43 44; do
        ACTIVE_RUN="iclr_primary_native_attn_cam_r16_s${ACTIVE_SEED}"
        rollout_signature="$(hash_text "$PIPELINE_SCHEMA|paper-test-rollout|$ACTIVE_DATASET|s$ACTIVE_SEED|$training_state_hash|$aid_manifest_hash|split=test")"
        run_step "08a_rollout_${ACTIVE_DATASET}_s${ACTIVE_SEED}" \
            "generate official test rollout for seed $ACTIVE_SEED on $ACTIVE_DATASET" \
            "$rollout_signature" paper_rollout_artifacts_ok generate_primary_test_rollout 1

        rollout_marker_hash="$(file_sha256 "$STATE_DIR/08a_rollout_${ACTIVE_DATASET}_s${ACTIVE_SEED}.done")"
        adapter_signature="$(hash_text "$PIPELINE_SCHEMA|standardize|$ACTIVE_DATASET|s$ACTIVE_SEED|$rollout_marker_hash|$aid_manifest_hash")"
        run_step "08b_standardize_${ACTIVE_DATASET}_s${ACTIVE_SEED}" \
            "standardize primary seed $ACTIVE_SEED on $ACTIVE_DATASET" \
            "$adapter_signature" standardized_artifacts_ok standardize_active_seed 1
    done

    standardization_hash="$(hash_text "$(file_sha256 "$STATE_DIR/08b_standardize_${ACTIVE_DATASET}_s42.done")|$(file_sha256 "$STATE_DIR/08b_standardize_${ACTIVE_DATASET}_s43.done")|$(file_sha256 "$STATE_DIR/08b_standardize_${ACTIVE_DATASET}_s44.done")")"
    eval_signature="$(hash_text "$PIPELINE_SCHEMA|evaluation|$ACTIVE_DATASET|primary|paper|$standardization_hash|${RSWM_I3D_CHECKPOINT}")"
    run_step "08c_evaluate_$ACTIVE_DATASET" "run official paper evaluation for $ACTIVE_DATASET" \
        "$eval_signature" evaluation_artifacts_ok evaluate_active_dataset 1

    evaluation_marker_hash="$(file_sha256 "$STATE_DIR/08c_evaluate_$ACTIVE_DATASET.done")"
    table_signature="$(hash_text "$PIPELINE_SCHEMA|tables|$ACTIVE_DATASET|$evaluation_marker_hash")"
    run_step "09_tables_$ACTIVE_DATASET" "export paper tables for $ACTIVE_DATASET" \
        "$table_signature" table_artifacts_ok export_active_tables 1
done

CURRENT_STEP="complete"
echo "PIPELINE_COMPLETE=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "dataset=$DATASET"
echo "official_splits=$SPLITS"
echo "experiments=$EXPERIMENTS_ROOT"
echo "paper_tables=$TABLE_ROOT"
echo "FINAL_RESULT=PASS"
