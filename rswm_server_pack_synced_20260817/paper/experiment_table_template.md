# RS-Roll Experiment Table Templates

All cells are intentionally empty. Populate them only from locked artifacts
whose dataset checksum, split checksum, checkpoint identity, seed, condition
route, and metric implementation are recorded in a run manifest.

## Reporting rules

1. Never replace `—` with an estimate, remembered value, or value from an older
   incompatible run.
2. Report `mean ± std` over training seeds for headline model comparisons and
   include the number of seeds.
3. Use the same test samples and generation-seed policy for paired comparisons.
4. Remove exactly one I2V condition frame before all future-frame fidelity
   metrics.
5. Write **FVD-proxy (R3D-18)** for the repository metric. Use **FVD-I3D** only
   for the separately recomputed standard extractor, with its checkpoint hash.
6. Include sample counts for every distribution metric and subgroup. A missing
   dependency/weight is `NOT RUN`, never zero.
7. Diagnostic baselines, oracle outputs, external methods, and RS-Roll variants
   must remain visibly distinct.
8. Ignore the staged YAML evaluation-cap difference (`native=400`,
   `drop/all=100`) for headline scoring: evaluate every method on the exact same
   locked sample IDs and print the common `n`.
9. Repository FID values are cross-method comparable only when every method is
   recomputed with the same Inception path and preprocessing.

## Table 1: Dataset and protocol audit

| Dataset tier | Role | Source ID / revision | Source images | Accepted rollouts | Split | Source-disjoint | T distribution (17/25/33) | Endpoint overlap max | Semantic-gate status | Artifact SHA-256 |
|---|---|---|---:|---:|---|---|---|---:|---|---|
| AID | train/val/hidden test | `blanchon/AID` / — | — | — | 70/10/20 | — | — / — / — | — | — | — |
| WHU-RS19 | test-only transfer | `jonathan-roberts1/WHU-RS19` / — | — | — | test-only | — | — / — / — | — | — | — |
| RESISC45 | test-only transfer | `blanchon/RESISC45` / — | — | — | test-only | — | — / — / — | — | — | — |

**Required checks before filling:** no source identity crosses AID splits; every
canonical rollout satisfies `0 <= endpoint_overlap < 0.1`; every clip has
`T in {17,25,33}`; official WHU/RESISC tiers contain no fallback source.

## Table 2: Main comparison on the AID hidden test

| Method | Method class | Camera input | Trainable params | Seeds | FID ↓ | LPIPS ↓ | FVD-proxy (R3D-18) ↓ | FVD-I3D ↓ | Net angular error ↓ | Valid trajectory ↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | diagnostic | none | 0 | deterministic | — | — | — | — | — | — |
| Deterministic translation | diagnostic | 2D commanded shift | 0 | deterministic | — | — | — | — | — | — |
| External image extrapolator A `[TODO]` | external | method-specific | — | — | — | — | — | — | — | — |
| External image/video method B `[TODO]` | external | method-specific | — | — | — | — | — | — | — | — |
| RS-Roll Drop | proposed config | none | — | — | — | — | — | — | — | — |
| RS-Roll Native | proposed config | Pluecker | — | — | — | — | — | — | — | — |

Caption must say that native and drop use different LoRA target scopes in the
current configs; this table compares complete systems, not one isolated factor.

## Table 3: Requested rank-16 configuration ablation

| Config | Role | Pose mode | LoRA targets | Rank / alpha | Trainable params | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Per-segment angle ↓ | Turn F1 ↑ | Path error ↓ |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `native_attn_cam_r16` | main | native | `attn_cam` | 16 / 16 | — | — | — | — | — | — | — |
| `drop_attn_only_r16` | comparison | drop | `attn_only` | 16 / 16 | — | — | — | — | — | — | — |
| `all_r16_ablation` | capacity ablation | drop | `all` | 16 / 16 | — | — | — | — | — | — | — |

## Table 4: Matched controls needed for causal attribution

| Pose mode | LoRA targets | Rank | Purpose | Implemented? | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Control metric |
|---|---|---:|---|---|---:|---:|---:|---:|
| native | `attn_only` | 16 | isolate pose effect at fixed attention target | [TODO] | — | — | — | — |
| drop | `attn_only` | 16 | matched no-pose reference | [IMPLEMENTED] | — | — | — | — |
| native | `attn_cam` | 16 | camera-aware main system | [IMPLEMENTED] | — | — | — | — |
| drop | `attn_cam` | 16 | isolate active pose route at fixed target list | [TODO] | — | — | — | — |

When pose is dropped, camera-injection layers are not exercised. Report this
fact if their LoRA parameters appear in a broad target list.

## Table 5: Horizon scaling

| Method | T | Number of paths | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Net angle ↓ | Segment angle ↓ | Turn F1 ↑ | Endpoint/path error ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drop r16 | 17 | — | — | — | — | — | — | — | — |
| Drop r16 | 25 | — | — | — | — | — | — | — | — |
| Drop r16 | 33 | — | — | — | — | — | — | — | — |
| Native r16 | 17 | — | — | — | — | — | — | — | — |
| Native r16 | 25 | — | — | — | — | — | — | — | — |
| Native r16 | 33 | — | — | — | — | — | — | — | — |

## Table 6: Endpoint-overlap analysis

Use non-overlapping bins within the canonical interval and report the exact
interval convention. Do not treat adjacent-frame overlap as endpoint overlap.

| Method | Endpoint-overlap bin | n | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Control metric | 95% CI |
|---|---|---:|---:|---:|---:|---:|---|
| Drop r16 | `[0.00, 0.02)` | — | — | — | — | — | — |
| Drop r16 | `[0.02, 0.05)` | — | — | — | — | — | — |
| Drop r16 | `[0.05, 0.10)` | — | — | — | — | — | — |
| Native r16 | `[0.00, 0.02)` | — | — | — | — | — | — |
| Native r16 | `[0.02, 0.05)` | — | — | — | — | — | — |
| Native r16 | `[0.05, 0.10)` | — | — | — | — | — | — |

The separate high-overlap learnability control should be reported in its own
table because it uses a different data profile.

## Table 7: Test-only domain transfer

| Train domain | Test domain | Method | n | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Net angle ↓ | Relative gap vs AID | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| AID | AID hidden test | Drop r16 | — | — | — | — | — | reference | [TODO] |
| AID | AID hidden test | Native r16 | — | — | — | — | — | reference | [TODO] |
| AID | WHU-RS19 | Drop r16 | — | — | — | — | — | — | [TODO] |
| AID | WHU-RS19 | Native r16 | — | — | — | — | — | — | [TODO] |
| AID | RESISC45 | Drop r16 | — | — | — | — | — | — | [TODO] |
| AID | RESISC45 | Native r16 | — | — | — | — | — | — | [TODO] |

## Table 8: Parameter and hardware efficiency

| Config | Rank | Targets | Trainable params | % of adapted experts | Peak allocated GiB | Peak reserved GiB | CPU peak GiB | Train hours | Inference sec/clip | Frames/sec | OOM/retry count |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drop r16 | 16 | `attn_only` | — | — | — | — | — | — | — | — | — |
| Native r16 | 16 | `attn_cam` | — | — | — | — | — | — | — | — | — |
| All r16 | 16 | `all` | — | — | — | — | — | — | — | — | — |

Hardware caption: one NVIDIA A800-SXM4-80GB; BF16; micro-batch 1;
gradient accumulation 4; gradient checkpointing enabled; inactive expert
offloaded. Replace only with values from the captured environment/run manifest.

## Table 9: Multi-seed uncertainty

| Paired comparison | Metric | Seed values | Mean difference | 95% bootstrap CI | Paired permutation p-value | Test trajectories | Interpretation |
|---|---|---|---:|---|---:|---:|---|
| Native − Drop | FID | — | — | — | — | — | — |
| Native − Drop | LPIPS | — | — | — | — | — | — |
| Native − Drop | FVD-proxy | — | — | — | — | — | — |
| Native − Drop | trajectory metric | — | — | — | — | — | — |

Do not use a significance threshold as a substitute for effect size and
uncertainty. If only three training seeds are feasible, pair seed-level summaries
with a trajectory-level paired bootstrap and state the different uncertainty
sources.

## Figure checklist

- **Figure 1:** task diagram: condition crop + continuous camera path → video
  rollout; show endpoint overlap definition.
- **Figure 2:** method: c2w/intrinsics → relative pose → Pluecker rays → camera
  injection in Wan blocks; distinguish frozen weights and LoRA parameters.
- **Figure 3:** benchmark construction with source-disjoint grouping and
  17/25/33-frame boundary deduplication.
- **Figure 4:** quality versus horizon and overlap, with uncertainty.
- **Figure 5:** fixed-sample qualitative comparison and failure cases.

Every qualitative panel must name dataset, sample ID, horizon, endpoint overlap,
trajectory, checkpoint, and generation seed.
