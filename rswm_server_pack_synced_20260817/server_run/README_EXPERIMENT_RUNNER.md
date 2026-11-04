# RSWM formal experiment runner

This layer starts the existing LoRA training and evaluation entries after the
automatic scheduler reaches `READY_FOR_TRAINING`. It does not implement or
change model, dataset, LoRA, camera, trajectory, checkpoint, or metric logic.

## Formal configurations

| Config | Pose arm | LoRA target | Rank | Intended role |
|---|---|---:|---:|---|
| `configs/native_attn_cam_r16.yaml` | `native` | `attn_cam` | 16 | headline experiment |
| `configs/drop_attn_only_r16.yaml` | `drop` | `attn_only` | 16 | camera ablation |
| `configs/all_r16_ablation.yaml` | `drop` | `all` | 16 | injection-location ablation |

The real `lora_remotesense/new/train.py` is epoch-based and saves a native
checkpoint every epoch. Therefore formal configs require `max_steps: null` and
`save_interval: 1`; unsupported semantics fail validation instead of being
silently approximated.

All server paths use `${RSWM_BASE}` and default to `/nix/rsroll_25828`. An
explicit environment value can relocate the complete server workspace without
editing YAML; unresolved variables fail before training. Absolute paths remain
supported for backward compatibility.

`SEM_ARM_MODE` is resolved directly from `dataset.pose_mode`: `native` stays
`native`, while both drop configurations stay `drop`. Neither the experiment
name nor a shell default is used to infer this scientific identity field.

## Dry-run

Dry-run parses the config, resolves `${RSWM_BASE}`, verifies the dataset, split,
model checkpoint, training/evaluation entries, and disk space, then prints the
exact train/evaluation argv. It exits before scheduler checks, `nvidia-smi`, CUDA
visibility changes, manifest writes, or training:

```bash
bash server_run/rswm_train_runner.sh \
  --config configs/native_attn_cam_r16.yaml \
  --dry-run
```

## Start a formal run

First confirm `server_run/rswm_auto_scheduler.sh` has recorded all five readiness
gates and `STATE=READY_FOR_TRAINING`. Then run:

```bash
bash server_run/rswm_train_runner.sh \
  --config configs/native_attn_cam_r16.yaml
```

To request a physical GPU explicitly:

```bash
bash server_run/rswm_train_runner.sh \
  --config configs/native_attn_cam_r16.yaml \
  --gpu 2
```

The runner validates the requested physical ID against inherited
`CUDA_VISIBLE_DEVICES`, reuses `rswm_gpu_selector.py --check-now`, acquires
non-blocking experiment/GPU locks, exposes exactly one GPU, and runs the strict
A800 preflight. A busy or unavailable GPU causes an immediate exit; there is no
unbounded wait and no process termination or GPU reset.

The A800 launch gate retains `min_free_gib: 70`. An idle 80 GiB A800 normally
reports roughly 79--80 GiB free; 70 GiB leaves room for driver/runtime
reservation while rejecting a device already carrying a material workload.
CPU-only mocks enforce: 75 GiB passes, 60 GiB fails, and 50 GiB fails.

## Resume

Resume only selects a complete native `checkpoint-epoch-*` directory containing
both LoRA adapters plus `training_state.pt`, `training_config.json`, and
`run_manifest.json`:

```bash
bash server_run/resume_training.sh \
  --experiment-id 20260815_native_attn_cam_r16
```

Use `--dry-run` to inspect the selected checkpoint and exact command. The
existing `train.py --resume_from` remains responsible for restoring adapters,
optimizer state, LR scheduler state, and the next epoch; checkpoint formats are
unchanged.

## Evaluation

Training invokes evaluation only after a successful native checkpoint exists.
Evaluation can also be run independently:

```bash
bash server_run/run_evaluation.sh \
  --experiment-id 20260815_native_attn_cam_r16
```

The existing `lora_remotesense/new/evaluate.py` writes its current benchmark
metrics, including FID, LPIPS, FVD proxy, and existing protocol fields, beneath
`experiments/<id>/results/`.

## Manifest example

```yaml
manifest_version: 1
experiment_id: "20260815_native_attn_cam_r16"
timestamp: "2026-08-15T01:23:45+08:00"
status: "TRAINING"
hostname: "a800-server"
git_commit: "<commit-or-NOT_AVAILABLE>"
runtime:
  python_path: "/nix/rsroll_25828/envs/rswm_compat_test/bin/python"
  python_version: "<detected>"
  torch_version: "<detected>"
  cuda_version: "<detected>"
gpu:
  requested: "auto"
  physical_gpu_id: "2"
  CUDA_VISIBLE_DEVICES: "2"
model:
  checkpoint: "/nix/rsroll_25828/models/lingbot-world-base-cam"
dataset:
  train: "/nix/rsroll_25828/work/rsroll_official/data/train_geo.h5"
  validation: "/nix/rsroll_25828/work/rsroll_official/official_splits.json"
lora:
  rank: 16
  target_modules: "attn_cam"
training:
  batch_size: 1
  lr: 0.0001
  steps: null
  num_epochs: 5
```

Manifest updates are atomic and use restrictive file permissions. The manifest
also stores exact argv arrays, paths, the config digest, attempts, resume source,
status, and failure diagnostics without requiring manual bookkeeping.
