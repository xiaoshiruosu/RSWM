# RS-Roll Benchmark Protocol

Version: `rsroll.benchmark_manifest/v2`  
Task: single remote-sensing image + continuous requested camera trajectory ->
continuous spatial video rollout.

This is a spatial rollout benchmark, not a claim of forecasting future temporal
events.  It evaluates whether a model can produce a viewpoint-conditioned rollout
from one observed remote-sensing image.

## Canonical input and output

Input:

```text
condition image
+ target trajectory
+ frozen benchmark metadata
```

Adapter output:

```text
T-frame rollout, canonical runtime tensor (T,H,W,C) uint8
```

Adapters may accept `[T,C,H,W]` at their boundary, but the shared evaluator converts
and validates it.  A single image is never silently expanded into a video.  Image
methods require an explicitly named recursive/per-waypoint adaptation.

## Frozen sample identity

Every sample in the formal manifest records:

- `sample_id`
- `source_dataset`, `source_image`, `source_split`
- `trajectory_id`, `trajectory_type`, and trajectory checksum
- `frame_count` and `condition_frame=0`
- H5 ground-truth rollout reference and checksum
- condition-image checksum and perceptual dHash
- distinct condition-image and required rollout resolutions
- `overlap_profile`
- endpoint, mean-future, and minimum-future overlap
- sample checksum

The manifest also records the complete H5 checksum, split checksum, resolved sample
order, generation seed, resize/crop protocol, and its own stable checksum.  All methods
in one comparison must consume the exact same manifest checksum.

Build a manifest with:

```bash
python server_run/baselines/build_benchmark_manifest.py \
  --h5 /path/to/test.h5 \
  --split /path/to/official_splits.json \
  --split-name test \
  --source-dataset AID \
  -o manifests/aid_test.json
```

Using `--allow-waypoint-interpolation`, `--skip-h5-hash`, or `--max-samples`
marks the artifact non-headline; those switches are diagnostic escape hatches only.
The frozen checksum is validated whenever the manifest is consumed, so post-build
edits fail closed.

## Same-sample rule

The paper main table must use exactly the same sample IDs for every method.  Different
sample counts, hidden truncation, or method-specific subsets are not comparable.
Per-sample failures are recorded and excluded only under an explicit success-rate gate;
they are never replaced with zeros or black videos.  The paper must report both `N`
and success rate.

## Future-only scoring

Frame `t=0` is the supplied condition image and is excluded from visual/video quality
scores:

```text
score frames 1 ... T-1
```

Trajectory recovery may retain `t=0` as the displacement origin.  This distinction is
intentional: trajectory following needs an origin, while visual fidelity must not be
inflated by a frame the model was given.

## Overlap protocol

For every rollout:

```text
O_t = overlap(condition view at t=0, view at t)
```

The complete curve is stored as `frame_0 ... frame_(T-1)`.  Reports include:

- endpoint overlap;
- mean future overlap;
- minimum future overlap;
- the frame-indexed overlap curve.

Adjacent overlap `overlap(frame t-1, frame t)` is stored separately as a continuity
audit.  Endpoint overlap below 0.1 does **not** imply that all adjacent frames—or all
future frames—have overlap below 0.1.

## Difficulty bins

`Dense`, `Medium`, `Sparse`, and `Extreme` are derived from empirical quartiles of
`mean_future_overlap` in the frozen manifest.  No thresholds are guessed before the
distribution is observed.  If quartiles are not strictly ordered, bin status is
`UNRESOLVED` and samples are not assigned a named bin.  `endpoint_overlap < 0.1` is
reported as a separate property, not used as a substitute for the whole rollout curve.

## Trajectory taxonomy

The manifest distinguishes `straight`, `turning`, and `multi-segment` base types and
can add `continuous-angle` and `long-horizon` tags.  Methods declare one of:

- native continuous trajectory support;
- native discrete direction support;
- adapted trajectory support;
- trajectory ignored.

The benchmark must not give an image model future GT or hidden geometry merely because
it lacks continuous-trajectory inputs.

## Metrics

Fidelity/perceptual metrics:

- SSIM and PSNR, future-only;
- LPIPS, future-only and optional when its offline checkpoint is available;
- FID over future frames only;
- `FVD_R3D_PROXY`: torchvision R3D-18 feature distance, internal proxy only;
- `FVD_I3D`: StyleGAN-V-compatible I3D feature protocol, stored separately.

FID, `FVD_R3D_PROXY`, and `FVD_I3D` require at least 32 complete clips before any
feature extractor is loaded. Optional metric values can only be written by the shared
evaluator; the unified runner does not accept manual metric overrides.

The I3D implementation is `server_run/recompute_fvd_i3d.py` and the standardized
evaluator can invoke the same extractor with `--i3d-checkpoint`.  It fixes feature
extractor identity/checksum, first-predicted-frame sampling, feature dimension, and
condition-frame exclusion.  The StyleGAN-V comparison repository documents numerical
alignment of its TorchScript I3D port with the original TensorFlow implementation:
https://github.com/universome/fvd-comparison

Trajectory/control metrics reuse the existing registration/flow implementation and
include the established net-direction, segment-angle, turn, waypoint, Fréchet, DTW,
flow-magnitude, and validity measures.  Paper-facing aliases include
`direction_error`, `endpoint_error`, and `registration_success_rate`.  Missing metrics
are `null`, never `0`.

## Geometry claim boundary

Registration, optical flow, Plücker conditioning, and viewpoint consistency do not
constitute explicit depth estimation, a cost volume, 3D Gaussian reconstruction, or a
reprojection-constrained 3D model.  Permitted descriptions are:

- viewpoint-conditioned spatial consistency;
- trajectory consistency;
- spatio-temporal consistency.

The code and paper must not claim an explicitly geometry-aware reconstruction model.

## Dataset tracks

Training: AID only.  
Test-only: AID test, WHU-RS19, RESISC45.

Before cross-dataset reporting, build frozen manifests and run:

```bash
python server_run/baselines/check_dataset_leakage.py \
  manifests/aid_test.json manifests/whurs19_test.json manifests/resisc45_test.json \
  -o reports/cross_dataset_leakage.json
```

Exact hashes, source-path collisions, dHash near duplicates, categories, image sizes,
resize/crop protocol, and checksums are reported.  dHash is a screening tool; flagged
pairs require review.

## Standard result directory

```text
results/<experiment_id>/
  manifest.json
  config.json
  baseline_card.json
  sample_metrics.jsonl
  aggregate_metrics.json
  videos/
  logs/
```

Legacy `scores.json`, `resolved_manifest.json`, and the single-file `frames.npz` archive
are retained for compatibility and low inode usage.

The table exporter revalidates manifest/card/result schemas, checks sample IDs and
row counts, and records the SHA-256 of every aggregate file it consumes.
