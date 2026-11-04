# ICLR 2000-sample full pipeline

`run_iclr_full_pipeline.sh` is the single resumable entry for the current AID
primary experiment. It orchestrates existing formal components; it does not change
the model, LoRA recipe, dataset protocol, split seed, generation seed, or metrics.

## Frozen scope

- Dataset build entry: `server_run/build_correct.sh` only.
- Canonical builder: `lora_remotesense/build_dataset_from_benchmark.py`.
- AID, `N=2000`, window `128x222`, target `480x832`, steps `9`.
- Frame counts `17/25/33`; overlap interval `[0.0, 0.1)`.
- `RSWM_SEMGATE=1`, a local CLIP snapshot, and fully offline Hugging Face mode.
- Primary LoRA: native pose, `attn_cam`, rank/alpha `16/16`, seeds `42/43/44`.
- Training split seed `42`; paper generation seed `42`.
- Paper test: the source-disjoint official test split, 400 samples. Training-time
  validation remains on `val`; test rollouts use a separate directory and cannot be
  consumed by training.
- Table scope: the three primary seeds. External baselines and cross-dataset runs are
  intentionally not fabricated or silently substituted.

## What each checkpoint does

1. Checks the local Python environment, base model, I3D checkpoint, and observed GPU
   inventory.
2. Checks the offline AID snapshot and the four disjoint source pools.
3. Loads the local CLIP snapshot with `local_files_only=True`.
4. Obtains `train_geo.h5`. With `--wait-for-data`, it only polls the externally
   running build and never launches another builder.
5. Runs structural/protocol validation, an independent offline SemanticGate recheck,
   and the duplicate/source-reuse gate.
6. Creates the source-disjoint official split and frozen 400-sample AID test manifest.
7. Runs the three primary configs through `rswm_train_runner.sh`, resuming a run only
   from a complete native checkpoint.
8. Generates separate official-test rollouts, standardizes the three registered
   primary result aliases, and calls `run_iclr_evaluation_a800.sh` for paper metrics.
9. Exports null-preserving paper CSVs plus `export_manifest.json`.

Every substep has its own append-only log and atomic `.done` checkpoint. A completed
checkpoint is skipped only when its signature and artifacts still verify. A stale
checkpoint fails closed; it is never silently rerun.

## Server preflight

Codex did not run the server pipeline and makes no claim that the server, GPU, files,
or tmux session currently exist. Confirm them in the server terminal first:

```bash
ls -ld /nix/rsroll_25828 \
  /nix/rsroll_25828/envs/rswm_compat_test \
  /nix/rsroll_25828/models/lingbot-world-base-cam
ls -ld "$RSWM_SEMGATE_MODEL"
test -z "${LOCAL_AID_DIR:-}" || ls -ld "$LOCAL_AID_DIR"
ls -l "$RSWM_I3D_CHECKPOINT"
nvidia-smi
tmux ls
tail -n 100 /nix/rsroll_25828/work/rsroll_official/logs/build_2000.log
```

`LOCAL_AID_DIR` is optional only when `blanchon/AID` is already present below the
configured offline Hugging Face hub cache. `RSWM_SEMGATE_MODEL` must be an absolute
local snapshot directory containing CLIP config, processor config, and weights.

## Read-only plan

From the unpacked package root:

```bash
export RSWM_BASE=/nix/rsroll_25828
export RSWM_PYTHON=$RSWM_BASE/envs/rswm_compat_test/bin/python
export RSWM_SEMGATE=1
export RSWM_SEMGATE_MODEL=/absolute/local/clip/snapshot
export RSWM_I3D_CHECKPOINT=/absolute/local/i3d_torchscript.pt

bash server_run/run_iclr_full_pipeline.sh --plan
```

Plan mode does not create files, inspect CUDA, build data, train, or evaluate.

## Attach to the currently running 2000-sample build

After the terminal checks above, run this in a dedicated tmux session. It will wait
until `train_geo.h5` passes the complete 2000-sample canonical validator; it will not
start `build_correct.sh` while waiting.

```bash
export RSWM_CONFIRM_FULL_PIPELINE=YES
export RSWM_CONFIRM_FORMAL_RUN=YES
export RSWM_CONFIRM_BASELINE_RUN=YES
export RSWM_CONFIRM_EVALUATION_RUN=YES

bash server_run/run_iclr_full_pipeline.sh --execute --wait-for-data
```

If no external build is running and the pipeline should own the build, omit
`--wait-for-data`. To pin an allocated physical GPU, add `--gpu ID`; `auto` uses the
existing scheduler-selected GPU and does not bypass its readiness gates.

## Resume and inspect

Resume by rerunning the exact same command. Do not delete a marker merely to force
progress; first inspect its step log and artifacts.

```bash
bash server_run/run_iclr_full_pipeline.sh --status
tail -n 200 /nix/rsroll_25828/work/rsroll_official/logs/full_pipeline/pipeline.log
cat /nix/rsroll_25828/work/rsroll_official/pipeline_state/last_failure.env
```

Primary outputs:

- dataset: `/nix/rsroll_25828/work/rsroll_official/data/train_geo.h5`
- split: `/nix/rsroll_25828/work/rsroll_official/official_splits.json`
- test manifest: `/nix/rsroll_25828/work/rsroll_official/manifests/aid_test.json`
- experiments: `/nix/rsroll_25828/work/rsroll_iclr_experiments/`
- standardized results: `/nix/rsroll_25828/work/rsroll_iclr_baselines/aid/`
- paper tables: `/nix/rsroll_25828/work/rsroll_iclr_paper_results/aid/`
