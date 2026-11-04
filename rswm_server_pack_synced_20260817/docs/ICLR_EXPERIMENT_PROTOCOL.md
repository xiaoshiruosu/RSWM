# Preregistered experiment protocol

## Research question

Can a trajectory-conditioned video diffusion model roll out spatially unobserved remote-sensing
content continuously, and does its advantage over a frozen video prior remain when the generated
view has little overlap with the single conditioning observation?

The benchmark models spatial camera rollout over a static scene. It does not model elapsed Earth
time or future land-cover dynamics.

## Units and splits

- Source image is the grouping unit for train/val/test. All samples connected by source identity
  or identical `full_scene` content stay in one split.
- Official ratios are 70/10/20. Test remains hidden from the training and local validation loaders.
- Training seeds are 42/43/44; split seed is fixed at 42.
- The unit for paired confidence intervals is a source/sample trajectory, not an individual frame.
  Frames within one rollout are correlated observations.

## Axes

1. Adaptation: frozen, LoRA target family, LoRA rank. Full fine-tuning is not reportable until a
   separately validated distributed recipe exists.
2. Condition overlap: overlap between the t=0 conditioning crop and each predicted crop.
3. Horizon: total clip length 17/25/33 (one condition frame plus 16/24/32 predictions).
4. Cross-domain: AID-trained weights evaluated without target-domain retraining on WHU-RS19 and
   RESISC45, after cross-dataset duplicate checks.

Adjacent-frame overlap is a continuity audit, not the condition-overlap axis.

## Metrics

- Fidelity: SSIM, PSNR, LPIPS, FID on predicted frames only.
- Video distribution: standard I3D FVD on the first 16 predicted frames, minimum 32 videos.
- Control/geometry proxies: endpoint direction, segment angle, turn precision/recall, waypoint
  error, Fréchet/DTW path shape, and generated-vs-GT 2D flow-vector consistency.

The proxy metrics do not establish explicit 3D geometry, depth, or multiview consistency.

## Reporting rules

- Main configuration: mean and standard deviation across seeds plus paired source-level bootstrap
  confidence intervals.
- Rank/target ablations: seed 42, clearly labeled single-seed secondary results.
- FID/FVD are omitted, not zero-filled, below the registered sample-size gate.
- A run with a mismatched/overridden manifest, lenient submission contract, legacy prompt schema,
  flow path estimator, or identity diagnostic route cannot enter the headline table.
- No configuration row is a result until the corresponding run, evaluation, manifests, and gates
  exist on disk.
