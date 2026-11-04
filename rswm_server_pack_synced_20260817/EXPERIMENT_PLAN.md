# RS-Roll Official Experiment Plan

This plan converts the current implementation into submission evidence. It does
not authorize training, data downloads, or GPU use. No training was started while
creating this document.

## Status vocabulary

- **[IMPLEMENTED]**: code/configuration exists.
- **[VERIFIED]**: checked by the stated test or report scope.
- **[TODO]**: official artifact/result is absent.

## Research questions linked to experiments

- **RQ1:** Can camera-conditioned video diffusion support low-overlap continuous
  remote-sensing rollouts?
- **RQ2:** How do the native, drop, and all rank-16 configurations compare?
- **RQ3:** How does performance change with rollout horizon and endpoint overlap?
- **RQ4:** Do AID-trained adapters transfer to WHU-RS19 and RESISC45 without
  target-domain training?
- **RQ5:** What quality/control/efficiency trade-off does camera-aware LoRA make
  relative to diagnostics and external generators?

## Global invariants

Every official run must preserve:

- canonical output size: `480x832`;
- source crop size: `128x222`;
- horizons: `T in {17,25,33}`;
- endpoint overlap: `0 <= overlap < 0.1`;
- condition frames removed exactly once before fidelity scoring;
- source-disjoint official split, fixed `split_seed=42`;
- training seed and generation seed recorded separately;
- AID-only adaptation for the generalization table;
- `FVD-proxy (R3D-18)` labeling unless the I3D recomputation is used;
- fail-closed source identity and no synthetic/alternate-dataset fallback in an
  official dataset tier.

## Phase 1 — Dataset preparation and release gates

### 1.1 AID canonical training/evaluation artifact

**Goal.** Build the official AID rollout corpus and source-disjoint split.

**[IMPLEMENTED]** Builder, semantic gate integration, provenance fields,
overlap enforcement, 4n+1 horizon contract, split generator, duplicate/leakage
checks, and HDF5 validators.

**[TODO] Required outputs**

- official 4,000-sample AID train H5 (or the preregistered final size);
- `official_splits.json` with 70/10/20 group assignment;
- source/near-duplicate/leakage reports;
- `overlap_audit.json`, including endpoint and adjacent-step statistics;
- horizon, angle, category, class, and source distributions;
- semantic-gate attempt/accept/reject report with frozen model/prompt/preprocess
  identity;
- file SHA-256 manifest.

**Gate A.** Proceed only if all clips have T=17/25/33, all canonical endpoint
overlaps lie in `[0,0.1)`, no source group crosses splits, no official sample
uses fallback data, and the semantic-gate audit is interpretable.

### 1.2 WHU-RS19 test-only artifact

**[VERIFIED]** Decoder, heterogeneous-size normalization, provenance, and
fail-closed behavior are covered by unit/synthetic-chain tests in the repository.

**[TODO] Required outputs**

- full server-side real-source integrity JSON (1005 decodes, 19 label IDs,
  RGB/source-size distribution);
- official rollout H5 and checksum;
- overlap/horizon/taxonomy/semantic reports;
- proof that every sample source is exactly
  `jonathan-roberts1/WHU-RS19`;
- test-only split manifest.

### 1.3 RESISC45 test-only artifact

**[VERIFIED]** Checker self-test and fail-closed code path.

**[TODO] Required outputs**

- real offline-source integrity report for `blanchon/RESISC45`;
- official rollout H5 and checksum;
- mapped-class coverage and per-pool counts;
- proof that no PatternNet/synthetic fallback entered the artifact;
- test-only split manifest.

**Gate B.** Do not run generalization evaluation until both target-domain
artifacts pass the same schema, frame, overlap, provenance, and duplicate gates.

## Phase 2 — LoRA training on AID

### 2.1 Pre-training safety gate

**[IMPLEMENTED]** Experiment runner, dry-run report, A800 profile, GPU selector,
preflight, scheduler, and smoke interfaces.

**Required sequence before formal training**

1. Run runner dry-runs for all three YAMLs and archive resolved paths/semantics.
2. Verify the exact model/data/split/checkpoint paths and free disk.
3. Run the full real GPU smoke: model load, LoRA attach with trainable parameters
   > 0, forward, loss, backward, and optimizer step.
4. Capture the A800 environment and dependency freeze.
5. Require an explicit human go/no-go after the above artifacts are reviewed.

### 2.2 Requested rank-16 configurations

| Order | Config | Pose mode | LoRA target | Role | Current status |
|---:|---|---|---|---|---|
| 1 | `configs/drop_attn_only_r16.yaml` | drop | `attn_only` | pose-dropped comparison | [IMPLEMENTED], training [TODO] |
| 2 | `configs/native_attn_cam_r16.yaml` | native | `attn_cam` | camera-aware main system | [IMPLEMENTED], training [TODO] |
| 3 | `configs/all_r16_ablation.yaml` | drop | `all` | broad-target ablation | [IMPLEMENTED], training [TODO] |

Run the main native and drop systems with at least seeds 42/43/44 while holding
the source-disjoint split fixed. The all-module ablation may begin with the
preregistered seed-42 budget, but multi-seed uncertainty is preferable if it is
used in a headline claim.

### 2.3 Per-run evidence

Archive for every run:

- resolved config and immutable run manifest;
- data, split, base-checkpoint, code, and environment hashes;
- LoRA matched module list and trainable-parameter audit for both experts;
- gradient-activity audit (especially camera targets under `drop`);
- loss/validation curves and selected-checkpoint rule;
- optimizer, BF16, offload, attention backend, and seed settings;
- peak GPU allocated/reserved memory, CPU RAM, wall time, throughput, and retry
  count;
- checkpoints and completion/failure stamps.

**Gate C.** A run is eligible for evaluation only if its manifest matches the
official dataset/split, both experts contain nonzero trainable LoRA parameters,
no forbidden backbone parameter is trainable, and the checkpoint selection rule
was fixed before hidden-test scoring.

### 2.4 Matched controls for attribution

**[TODO, high priority]** The requested three configs do not isolate pose input
from LoRA target scope. Add a minimal crossed design:

- native + `attn_only`, rank 16;
- drop + `attn_cam`, rank 16;

with otherwise identical training settings. If these controls cannot be run,
the paper must compare “complete configurations” and remove component-level
causal language.

## Phase 3 — Baselines

### 3.1 Diagnostic baselines

**[IMPLEMENTED]** Persistence, deterministic translation, random direction,
and GT oracle utilities.

Use persistence/translation/random direction to validate that the benchmark and
control metrics behave as expected. Keep GT oracle labeled as an upper-bound
instrument, never as a deployable method.

### 3.2 External generative baselines

**[TODO] Minimum submission set**

1. A remote-sensing/image spatial-extrapolation baseline compatible with a
   single observed crop.
2. A strong image-to-video or camera-controlled video generation baseline.
3. Optionally a recursive outpainting baseline to test whether video modeling
   adds more than repeated image extension.

For every external baseline, freeze:

- paper/code version and checkpoint checksum;
- allowed inputs and any trajectory conversion;
- output frame count/resolution and no-access-to-target rule;
- inference seed/sampling settings;
- adapter code and runtime environment;
- the same hidden samples and scoring path used by RS-Roll.

**Gate D.** No “better than baselines” sentence is allowed until at least one
credible external generator has successful, protocol-matched outputs. Diagnostic
baselines alone do not satisfy this gate.

## Phase 4 — Evaluation

### 4.1 Metric validation

- Run known-answer tests and archive versions of Inception, LPIPS, R3D-18, and
  I3D weights.
- Calibrate direction/path metrics on ground truth and deterministic transforms.
- Record undefined/near-static cases rather than forcing an angle.
- Enforce minimum sample counts for FID/FVD and every subgroup.

### 4.2 AID hidden-test evaluation

For each eligible checkpoint and baseline:

1. Generate the complete fixed hidden-test set.
2. Validate sample IDs, T, H/W, dtype/range, condition-frame placement, and
   manifest identity.
3. Override the staged per-config evaluation caps (`native=400`,
   `drop/all=100`) with one common locked list of sample IDs.
4. Score FID, LPIPS, FVD-proxy, trajectory metrics, and efficiency.
5. Recompute standard I3D-FVD for headline/cross-paper comparison.
6. Aggregate native/drop across training seeds with paired uncertainty.

### 4.3 Diagnostic breakdowns

Report results by:

- horizon: 17/25/33;
- endpoint-overlap bins within `[0,0.1)`;
- scene category and source class (with n);
- trajectory segment count, heading, and turn magnitude;
- accepted semantic-gate strata where scientifically meaningful.

Keep the high-overlap learnability control separate from the canonical table.

### 4.4 Test-only domain transfer

Evaluate the same AID-trained checkpoints on WHU-RS19 and RESISC45 with no
target-domain tuning. Report absolute values and the paired gap relative to AID.
Run cross-dataset duplicate checks before interpreting transfer.

### 4.5 Qualitative and human audit

Use a fixed selection rule to show median, best, worst, and representative
failure cases at each horizon. Include command path overlays and target frames.
If a human study is added, preregister questions, randomization, annotator count,
and agreement analysis.

**Gate E.** Results enter the paper only after sample-contract checks, metric
identity checks, and manifest joins pass. Any missing metric is shown as
`NOT RUN`/`N/A`, never replaced by zero.

## Phase 5 — Paper tables and claim freeze

1. Populate `paper/experiment_table_template.md` directly from locked JSON/CSV
   artifacts.
2. Generate the main, ablation, horizon, overlap, transfer, and efficiency
   tables from one results registry.
3. Fill the abstract result sentence last, using effect sizes and uncertainty.
4. Replace each `[TODO]` only after linking it to an artifact in the claim-evidence
   map.
5. Run a claim audit: every number has dataset, n, seed, metric implementation,
   and checkpoint identity.
6. Freeze qualitative sample IDs before polishing figures.
7. Conduct an internal reviewer pass using `paper/reviewer_questions.md`.

## Priority order

### P0 — blocks any scientific result

1. Official AID artifact + source-disjoint/overlap/semantic audits.
2. Real full-chain A800 smoke and environment capture.
3. Multi-seed native/drop training with complete manifests.

### P1 — blocks a defensible ICLR comparison

4. Matched pose × target ablation.
5. At least two credible external baseline families under the same protocol.
6. Standard I3D-FVD and calibrated trajectory metrics.
7. WHU-RS19 and RESISC45 test-only transfer.

### P2 — strengthens analysis and rebuttal readiness

8. Horizon/overlap/category breakdowns.
9. Parameter, memory, and throughput table.
10. Fixed-sample failure analysis and optional human evaluation.

## Stop conditions

Stop and diagnose rather than continuing if any of the following occurs:

- official dataset identity, checksum, split, or overlap audit changes;
- a run resolves the wrong `pose_mode`, target modules, rank, or dataset path;
- trainable LoRA parameters are zero or non-LoRA backbone parameters train;
- a target-domain sample is used for adaptation;
- GPU smoke cannot complete backward and optimizer step;
- metric code silently changes shape, frame count, condition-frame, or extractor;
- external baseline receives privileged inputs unavailable to RS-Roll;
- results exist without a manifest that proves their identity.

Negative experimental outcomes are not a stop condition. They should be
reported and used to refine the paper's claim.
