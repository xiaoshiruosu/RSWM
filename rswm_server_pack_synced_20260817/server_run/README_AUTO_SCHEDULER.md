# Shared A800 unattended readiness scheduler

`rswm_auto_scheduler.sh` waits for the official RESISC45 cache, runs all
CPU-only data checks and two 12-sample real builder smokes, then waits for one
allowed A800 to remain idle. It runs the existing one-step LoRA A800 smoke and
stops at `READY_FOR_TRAINING`; it never starts formal training.

The three Hugging Face offline variables remain unset during `STARTING` and
`WAITING_DATA`. They are enabled only after the RESISC45 cache, `refs/main`,
resolved snapshot, zero `.incomplete` files, and absence of the download
process have all been confirmed. The transition is recorded as
`OFFLINE_MODE_ENABLED`.

The builder's existing four-scene balance requires the sample count to be a
multiple of four. Therefore 12 is the smallest valid smoke size not below the
requested ten samples. This does not change any formal dataset or training
configuration.

## Start in tmux

From the unpacked package root:

```bash
tmux new-session -A -s rswm_scheduler
bash server_run/rswm_auto_scheduler.sh
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t rswm_scheduler
```

Query status without changing any process:

```bash
bash server_run/rswm_scheduler_status.sh
```

## Dry-run (no GPU and no training)

Use the dry-run before the unattended launch to verify the state file, log,
single-instance lock, and the real GPU-selector code against a generated
`nvidia-smi` mock:

```bash
bash server_run/rswm_auto_scheduler_dryrun.sh
```

It simulates `WAITING_DATA`, `WAITING_GPU`, and `GPU_SELECTED`, exports an empty
`CUDA_VISIBLE_DEVICES`, never calls the system `nvidia-smi`, and ends at
`DRYRUN_COMPLETE`. Its state and lock are separate from the production
scheduler:

```text
/nix/rsroll_25828/work/rswm_scheduler_dryrun_status.env
/nix/rsroll_25828/work/rswm_auto_scheduler_dryrun.lock
```

`RSWM_DRYRUN_BASE` may point the dry-run artifacts at an isolated test
directory; it does not change the production scheduler base path.

Only one scheduler instance can pass the project lock at
`/nix/rsroll_25828/work/rswm_auto_scheduler.lock`. This lock does not reserve
a GPU against other users; all GPU decisions are made from fresh `nvidia-smi`
metrics and compute-process checks.

## Safe overrides

Defaults:

```bash
DATA_POLL_SECONDS=300
GPU_IDLE_CONFIRM_SECONDS=600
GPU_POLL_SECONDS=60
GPU_MAX_UTIL=10
GPU_MAX_USED_MB=5000
RSWM_PYTHON=/nix/rsroll_25828/envs/rswm_compat_test/bin/python
```

Example with a five-minute confirmation window:

```bash
GPU_IDLE_CONFIRM_SECONDS=300 \
GPU_MAX_USED_MB=3000 \
bash server_run/rswm_auto_scheduler.sh
```

If `CUDA_VISIBLE_DEVICES` is already set to numeric physical IDs, the scheduler
only considers that subset. After selecting (for example) physical GPU 2, it
exports `CUDA_VISIBLE_DEVICES=2`; Python correctly sees that device as logical
GPU 0. Logs explicitly record physical ID, visibility, and process-local ID.

## State and outputs

The atomic state file is:

```text
/nix/rsroll_25828/work/rswm_scheduler_status.env
```

The public state sequence is:

```text
STARTING
→ WAITING_DATA
→ CPU_TEST
→ WAITING_GPU
→ CONFIRMING_GPU
→ GPU_PREFLIGHT
→ DATASET_SMOKE
→ GPU_SMOKE
→ READY_FOR_TRAINING
```

Logs are written to
`/nix/rsroll_25828/logs/auto_scheduler_YYYYMMDD_HHMMSS.log`. Builder-smoke H5
files and JSON checks are retained under a timestamped directory inside
`/nix/rsroll_25828/work/`; the GPU smoke result marker is under
`/nix/rsroll_25828/work/a800_smoke/`.

Any failed command moves the state to `FAILED` and records the failed step,
exit code, command, timestamp, and log path. A candidate GPU that becomes busy
or gains a new compute process is abandoned without signalling that process;
the scheduler returns to `WAITING_GPU`.
