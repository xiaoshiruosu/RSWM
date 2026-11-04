# Remaining Experiments

Everything below is **NOT RUN** unless a later signed result manifest says otherwise.

## Gate 1: real A800 smoke

1. Stage the exact base checkpoint, AID H5, frozen split, and offline metric assets.
2. Run the structured checks in `GPU_SMOKE_TEST.md`.
3. Confirm model load, LoRA injection, trainable parameter count, forward, loss,
   backward, optimizer step, dtype, and peak memory.
4. Do not start the matrix if any item fails.

## Gate 2: freeze benchmark manifests

Build AID test, WHU-RS19 test, and RESISC45 test v2 manifests.  Verify checksums,
trajectory arrays, overlap curves, empirical bins, image sizes, and source provenance.
Run cross-dataset exact/near-duplicate audit and manually review every dHash flag.

## Gate 3: lower bounds and pretrained baselines

- Evaluate Persistence on the full frozen manifests.
- Run frozen pretrained I2V with zero adaptation parameters.
- Run same-backbone no-camera under the matched recipe.
- Stage and GPU-smoke the SD2 inpainting checkpoint, then evaluate the adapted rollout.
- Decide whether the unavailable RSWISE/RemoteBAGEL implementation can be obtained.
  If not, keep its result cells blank and report the reproducibility blocker.
- Keep MVSplat360 in a separate multi-view track if run.

## Gate 4: matched 2x2 and primary run

Train/evaluate the four rank-16 matched cells:

- native + attn_only;
- native + attn_cam;
- drop + attn_only;
- drop + attn_cam.

Use identical training data, split, steps/epochs, batch/accumulation, test manifest,
frame protocol, and seed protocol.  Run seeds 42/43/44 for the declared primary arm.

## Gate 5: rank scaling

Train `native + attn_cam` at ranks 4, 8, 16, 32, and 64.  Record:

- trainable parameters;
- peak training memory;
- training and inference time;
- FID, LPIPS, `FVD_R3D_PROXY`, `FVD_I3D`;
- trajectory metrics.

Unavailable values remain null.

## Gate 6: final evaluation/export

- Run future-only evaluation for every method/dataset pair.
- Compute standard I3D-FVD only with the frozen extractor/checksum and at least the
  declared distribution-metric sample minimum.
- Verify success rate and exact sample identity.
- Export all CSV tables with `tools/export_paper_tables.py`.
- Inspect qualitative rollouts, failed samples, overlap curves, and trajectory types.
- Only then update lifecycle flags `EVALUATED` and `RESULT_AVAILABLE`.

## External blockers

1. RemoteBAGEL/RSWISE: no verified official runnable code/checkpoint found.
2. SD2 inpainting: real offline checkpoint absent locally.
3. MVSplat360: upstream repo, separate environment, and checkpoints absent locally.
4. Standard I3D-FVD: real TorchScript asset not present in this local source tree.
5. All LoRA/frozen formal results: no checkpoint/result artifacts bundled.

