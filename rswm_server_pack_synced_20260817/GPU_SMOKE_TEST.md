# A800 GPU Smoke Test

Target: NVIDIA A800-SXM4-80GB.  This checklist is for the formal server; it was **NOT
RUN** while preparing the baseline extension.

## Important distinction

`python server_run/a800_smoke.py --level 3` validates a synthetic BF16 LoRA
forward/backward/AdamW and optimizer-state CPU/GPU round trip.  It is useful, but it
does not load the real LingBot-World checkpoint.  Passing it alone is insufficient.

## Required sequence

1. Record `nvidia-smi`, driver, CUDA, PyTorch, Python, and free disk.
2. Run the synthetic infrastructure smoke:

   ```bash
   python server_run/a800_smoke.py --level 3
   ```

3. Dry-run the real experiment config and verify all paths without allocating a GPU:

   ```bash
   bash server_run/rswm_train_runner.sh \
     --config configs/benchmark_ablation/native_attn_cam_r16.yaml \
     --dry-run
   ```

4. On the allocated A800, execute a dedicated one-batch/one-step probe using the same
   base checkpoint, dataset schema, pose route, LoRA injection code, precision, and
   optimizer as the formal run.  It must perform, in order:

   ```text
   load base checkpoint
   construct Video DiT + VAE/text components
   inject LoRA
   assert trainable LoRA parameters > 0
   forward
   finite loss
   backward
   assert finite/non-zero LoRA gradients
   optimizer.step()
   optimizer.zero_grad()
   ```

5. Stop after the probe.  Do not transition to formal training automatically.

## Mandatory structured log

```text
SMOKE RESULT
model_load: PASS/FAIL
lora_attach: PASS/FAIL
forward: PASS/FAIL
loss: <finite scalar>
backward: PASS/FAIL
optimizer_step: PASS/FAIL
total_parameters: <integer>
trainable_parameters: <integer > 0>
trainable_percent: <float>
lora_module_names: <complete or archived list>
gpu_name: NVIDIA A800-SXM4-80GB
gpu_peak_allocated_gib: <float>
gpu_peak_reserved_gib: <float>
dtype: <actual dtype>
pose_mode: native
lora_rank: 16
target_modules: attn_cam
checkpoint_sha256_or_revision: <value>
dataset_checksum: <value>
TRAINING_STARTED: 0
```

## Hard failure conditions

- zero trainable LoRA parameters;
- base/checkpoint identity mismatch;
- wrong pose route or target-module set;
- NaN/Inf loss or gradients;
- optimizer step does not change at least one LoRA parameter;
- OOM or peak memory above the operational safety margin;
- missing parameter/module/memory/dtype log fields;
- any automatic continuation into formal training.

Only after a human reviews the log may the formal matrix be scheduled.

