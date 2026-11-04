# ICLR main experiments on A800 80GB

For the current resumable AID `N=2000` build -> three-primary-seed -> official-test
evaluation -> paper-table entry, see
`server_run/README_ICLR_FULL_PIPELINE.md` and
`server_run/run_iclr_full_pipeline.sh`. The wrappers below remain the individual
formal components used by that orchestrator.

This directory is an orchestration layer over the existing formal runner and fair
baseline platform. It does not change LingBot/Wan Video DiT, the VAE, camera or
Pluecker conditioning, trajectory construction, LoRA injection, dataset loaders, or
metric implementations.

## Safety model

Every wrapper defaults to `--plan`. Plan mode validates the machine-readable matrix
and prints safe commands ending in the underlying runner's dry-run option. It does
not probe CUDA or start training. Server dry-run performs real path/config checks but
still passes `--dry-run` to the formal runner. Formal execution requires an explicit
environment confirmation and retains the scheduler readiness gates, GPU lock,
A800 preflight, level-3 smoke prerequisite, offline mode, and manifest lifecycle in
`server_run/rswm_train_runner.sh`.

## 1. Inspect the frozen plan (CPU only)

```bash
python server_run/iclr_a800/iclr_main_plan.py validate
python server_run/iclr_a800/iclr_main_plan.py list-runs --group primary
bash server_run/iclr_a800/run_iclr_main_a800.sh --plan --group primary
```

Primary runs are `native + attn_cam + rank 16` with training seeds 42/43/44.
The data split and generation seed remain fixed at 42. The matched 2x2 and rank
sweep reuse the seed-42 primary cell instead of retraining an identical point.

## 2. A800 server dry-run (no GPU)

```bash
export RSWM_BASE=/nix/rsroll_25828
export RSWM_PYTHON=$RSWM_BASE/envs/rswm_compat_test/bin/python
bash server_run/iclr_a800/run_iclr_main_a800.sh \
  --server-dry-run --group primary
```

This checks the real checkpoint/data/split paths and prints the exact training and
evaluation argv through the existing runner. It does not inspect CUDA.

## 3. Formal training (explicit opt-in)

Start with one run, then expand to the complete primary group only after its manifest
and output are reviewed:

```bash
export RSWM_CONFIRM_FORMAL_RUN=YES
bash server_run/iclr_a800/run_iclr_main_a800.sh \
  --execute --group primary \
  --run iclr_primary_native_attn_cam_r16_s42
```

The wrapper does not bypass `READY_FOR_TRAINING`; the existing scheduler must have
passed DATA READY, CPU TEST, GPU PREFLIGHT, DATASET SMOKE, and GPU SMOKE. On a shared
server, pass `--gpu ID` only for a GPU allocated to you. Without it, the formal
runner uses the scheduler-selected physical GPU.

For an interrupted run, keep the same experiment id and add `--resume --run ID`;
the formal runner accepts only a complete checkpoint and records a new attempt in the
existing experiment manifest.

Available training groups:

| Group | Runs | Purpose |
|---|---:|---|
| `primary` | 3 | rank-16 camera-aware mean/std over seeds 42/43/44 |
| `matched` | 4 | native/drop x attn_only/attn_cam at rank 16 |
| `target_all` | 2 | optional native/drop all-target ablation |
| `rank` | 5 | rank 4/8/16/32/64 with native+attn_cam fixed |
| `all` | 12 unique | union of every canonical run above |

## 4. Unified baselines

The CPU persistence lower bound can be inspected independently:

```bash
bash server_run/iclr_a800/run_iclr_baselines_a800.sh \
  --plan --dataset aid --group lower_bound
```

The `main` baseline group also lists frozen I2V, image outpainting, the explicitly
adapted RSWISE bridge, same-backbone no-camera, and RS-Roll. Their local checkpoint or
producer-output environment variables must exist before `--execute`; missing assets
are fatal and no result is fabricated. Cross-dataset calls change only the frozen
manifest (`whurs19` or `resisc45`) and never train on those datasets.

The three primary seeds are separate aliases (`rsroll_primary_s42/s43/s44`) that all
use the same registered RS-Roll adapter but point to distinct formal evaluation
directories. Stage them explicitly, for example:

```bash
export RSWM_NATIVE_ATTN_CAM_R16_S42_EVAL_DIR=$RSWM_BASE/work/rsroll_iclr_experiments/iclr_primary_native_attn_cam_r16_s42/results
export RSWM_NATIVE_ATTN_CAM_R16_S43_EVAL_DIR=$RSWM_BASE/work/rsroll_iclr_experiments/iclr_primary_native_attn_cam_r16_s43/results
export RSWM_NATIVE_ATTN_CAM_R16_S44_EVAL_DIR=$RSWM_BASE/work/rsroll_iclr_experiments/iclr_primary_native_attn_cam_r16_s44/results
```

## 5. Future-only evaluation and tables

```bash
bash server_run/iclr_a800/run_iclr_evaluation_a800.sh \
  --plan --dataset aid --group main --metrics paper
```

`core` computes CPU SSIM/PSNR. `paper` additionally requests LPIPS, FID,
`FVD_R3D_PROXY`, and standard `FVD_I3D`; it requires a local
`RSWM_I3D_CHECKPOINT`. Frame 0 is excluded in both modes. The final command uses the
existing fail-closed paper-table exporter.

## Status

The configuration and CPU validation paths are implemented. No formal training,
baseline generation, A800 smoke, or paper metric was run while preparing these
scripts. Until real standardized result directories exist, paper values remain
`NOT RUN` rather than zero or placeholders.
