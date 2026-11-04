# Original Wan on one A100 40GB: CPU block offload

Software version 0.4.1 provides a real autograd-compatible offload path. It does
not certify that the full pretrained model fits a 40GB GPU. Measure complete
native inference, training, checkpoint recovery and adapter inference on the
allocated server before claiming engineering acceptance.

## Explicit configuration

Keep the actual reviewed prompt, negative prompt, steps, shift, guidance,
boundary, camera configuration, data selection, frame count, dimensions and
seed. Add the following settings to the existing locked backend configuration:

```json
{
  "sampling": {
    "param_dtype": "bf16",
    "t5_dtype": "bf16",
    "t5_cpu": true,
    "inactive_model_cpu": true,
    "block_cpu_offload": true
  },
  "training": {
    "expert_residency": "block_cpu",
    "trainable_policy": {"gradient_checkpointing": true}
  }
}
```

This is a fragment, not a runnable complete configuration or an approval.
Training still requires all existing rank/alpha/target/mode, uniform timestep,
precision and finite dev-validation fields. Frozen inference uses
`training=null` and the same explicit `sampling.block_cpu_offload=true`.
Prepare a **new** source tree with the updated `scripts/prepare_legacy_wan.py`;
the old prepared sampler does not implement this flag. Retain previous source
trees, configuration and receipts. New hashes must come from actual bytes.

## Execution and numerical behavior

Both original camera experts load on CPU. Each directly owned frozen tensor is
staged for its module and returned to its CPU master storage after use. LoRA
parameters remain on the execution device without changing Parameter identity
or optimizer bindings. Frozen canonical Parameters remain on CPU; module
calculation slots temporarily expose independent staged Tensors and restore
the registered Parameter objects at call boundaries. This prevents PyTorch's
non-reentrant recomputation cache from retaining an object whose device changes
after forward. Version 0.4.0 failed this real A100 checkpoint case (retained
job 423050); its failure is not reclassified as a pass. Saved-tensor hooks move autograd storage to CPU and
restore tensors when backward requires them. Original non-reentrant activation
checkpointing remains enabled; shared saved tensors use weak CPU-snapshot
deduplication, without retaining GPU copies.

T5 encodes on CPU during training. VAE encode runs as a separate stage before
expert forwards. The inference sampler bypasses whole-expert `.to(cuda)` when
block offload is explicitly selected. Frame count, spatial grid, original flow
velocity objective, per-sample low/high routing and all validation denominators
remain unchanged. Quantization is not used.

Original Wan explicitly computes time embedding, time projection, the output
head and affine `norm3` in FP32. The strict loader keeps those original
checkpoint tensors in FP32 **at loading**, while DiT blocks use BF16. Casting an
already rounded BF16 tensor back to FP32 does not restore checkpoint precision.
Small real vendor-model tests reproduce the previous full-BF16 type error and
verify the corrected loading/forward/backward behavior.

Eviction compares frozen weight bytes with CPU masters, including signed-zero
changes; a `.data` mutation is a failure, not silently reverted evidence.
Forward exceptions and checkpoint recomputation early stops release staged
weights. Checkpoint/hash/export boundaries quiesce the controllers; resume
loads the original full driver checkpoint on CPU and keeps strict tensor keys,
frozen identities, RNG, optimizer and cumulative validation costs.

## Shared-server resources and acceptance

Use one Slurm-allocated GPU and bounded CPU/RAM/time. Offload transfers work to
host RAM and PCIe; it does not eliminate memory or transfer costs. The two
experts alone need about 70 GiB of host parameter storage. Full checkpoint
restore may temporarily retain another full state. A real-model trial needs a
measured host-RAM budget, initially in the 192–256 GiB range, rather than the
8 GiB used for source imports. This is a planning estimate, not a RAM guarantee.

Training loss events retain real process-lifetime GPU allocated/reserved peaks,
host peak RSS, and controller transfer counters, including failure paths.
Allocator peaks are not replaced by post-cleanup current allocation. CUDA
tests skipped on a CPU host are not GPU passes. Small synthetic CPU numerical
equivalence, including actual vendor modules and PEFT, is software evidence
only; pretrained A10040 execution remains required.

The older core120 build/audit and successful recovered Python environment are
retained. Do not download or materialize those sources again. If a downstream
stage requires a current source fingerprint, independently audit the retained
build with the new code and retain the old audit as historical evidence.

Weight download is a separate CPU job pinned to original camera-model revision
`6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd`, 26 files,
160246219231 bytes. Download and hash verification do not approve a backend or
establish real-model/research acceptance.

## Server training entry in 0.4.2

The server training-source stage now carries the explicit
sampling.block_cpu_offload selection into expert_residency=block_cpu.
Absent/false keeps the existing resident mode. The derived policy must pass
the same original backend validation, including CPU placement and activation
checkpointing. A pre-existing non-null training policy must match the
project/spec-derived frozen policy; it is never silently overwritten.

Owner job 423053 passed all three small CUDA regressions on A100 40GB with
software 0.4.1 (source a157b58c0d67b93ad5e89301ebe1f8c2731b84746ae3efba0fc35a011c9b5ff2).
That is retained historical evidence for unchanged offload implementation
bytes. It does not prove full pretrained inference, training, recovery, or
research acceptance for either version.
