# RS-Roll: Research Questions, Novelty, and Evidence Map

This document is the supervisor-style claim blueprint for the ICLR draft. It is
not a results report. Status labels have the following meanings:

- **[IMPLEMENTED]**: the relevant code/configuration exists and was inspected.
- **[VERIFIED]**: the stated property is supported by a checked test or repository
  report. The verification scope is stated explicitly.
- **[TODO]**: empirical evidence is still missing. A TODO item must not be written
  as a completed result in the paper.

## Candidate titles (not finalized)

1. **Camera-Aware Adaptation of Video Diffusion World Models for Low-Overlap Remote-Sensing Extrapolation**
2. **RS-Roll: Continuous Low-Overlap Spatial Rollouts from a Remote-Sensing Image**
3. **Toward Remote-Sensing World Models with Trajectory-Conditioned Video Rollouts**
4. **Learning Continuous Spatial Extrapolation with Camera-Conditioned Video Diffusion**
5. **RSWM-Bench: Evaluating Continuous Low-Overlap Video Rollouts in Remote Sensing**

Candidates 1--4 emphasize the method-plus-task paper; Candidate 5 is appropriate
only if the benchmark becomes the dominant validated contribution.

## Paper type and scope

The intended paper type is **new problem setting + benchmark protocol +
parameter-efficient adaptation study**. It is not a claim that RS-Roll invents
video diffusion, explicit 3D reconstruction, or a new LoRA algorithm.

The spatial rollout is a sequence of crops along a trajectory over a static
remote-sensing scene. “Video” denotes the ordered rollout representation. It
must not be described as forecasting physical Earth changes through time.

## Primary research question

> **RQ1.** Can a camera-conditioned image-to-video diffusion world model be
> adapted parameter-efficiently to generate continuous remote-sensing spatial
> rollouts when the conditioning view and the terminal target view have less
> than 10% spatial overlap?

### Secondary research questions

- **RQ2 (conditioning):** Under a controlled evaluation protocol, how does the
  `native + attn_cam` configuration compare with a pose-dropped,
  attention-only configuration on rollout fidelity and trajectory adherence?
- **RQ3 (adaptation scope):** Does widening rank-16 LoRA from attention-only to
  all transformer-block linear modules improve quality enough to justify its
  additional trainable capacity?
- **RQ4 (horizon):** How do errors change across 17-, 25-, and 33-frame rollouts
  and across overlap/horizon strata?
- **RQ5 (generalization):** Do AID-trained adapters transfer without training on
  WHU-RS19 and RESISC45 under the same construction and evaluation protocol?

## Central hypothesis

**[TODO]** Camera-coordinate conditioning should provide a stronger inductive
bias for trajectory-following spatial rollouts than text/direction conditioning
alone, while the pretrained video model supplies temporal continuity priors.
This is a hypothesis to be tested, not a result already established by the code.

The implementation uses Pluecker ray embeddings derived from camera-to-world
poses and intrinsics. This is coordinate-aware conditioning; it is not evidence
that the model reconstructs explicit metric 3D geometry.

## Novelty boundary

### What is plausibly novel

1. **[IMPLEMENTED] Problem formulation.** Remote-sensing spatial extrapolation
   is formulated as a trajectory-conditioned video rollout rather than a
   one-step adjacent-tile image generation problem.
2. **[IMPLEMENTED] Evaluation setting.** The canonical benchmark uses continuous
   multi-segment directions, random start positions, source-disjoint splitting,
   17/25/33-frame horizons, and a hard start-to-end overlap interval
   \([0,0.1)\).
3. **[IMPLEMENTED] Domain adaptation design.** Rank-16 LoRA is attached to a
   camera-enabled LingBot-World/Wan I2V backbone with explicit `attn_cam`,
   `attn_only`, and `all` target scopes.
4. **[IMPLEMENTED] Evidence-oriented protocol.** The repository distinguishes
   visual fidelity, perceptual/video distribution metrics, trajectory-control
   diagnostics, source leakage checks, and test-only cross-dataset evaluation.

### What is not claimed

- A new video diffusion architecture, Video DiT, VAE, 3D RoPE, Pluecker
  representation, CLIP model, or LoRA algorithm.
- Explicit scene geometry recovery or metrically correct 3D reconstruction.
- Physical or temporal forecasting of Earth dynamics.
- State-of-the-art performance, superior baselines, or completed cross-domain
  generalization before the missing experiments are run.

## Limitation-to-design logic

| Prior limitation / evaluation gap | Design challenge | Implemented response | Evidence needed for the paper |
|---|---|---|---|
| Adjacent-tile image extrapolation is often discretely directed and single-step | Represent arbitrary, changing motion over a rollout | Continuous-angle, 2--4-segment trajectories with 9 samples per segment and deduplicated boundaries | Distribution audit of angles/turns; per-segment control metrics |
| Large conditioning overlap can reward copying | Force generation into mostly unseen terminal regions | Hard start/end crop-overlap constraint \(0\leq\Omega<0.1\) | Dataset audit showing every official sample satisfies the interval |
| Image generators do not directly encode camera-coordinate control | Connect rollout coordinates to a pretrained video model | Nadir, planar c2w trajectories; relative poses; Pluecker rays; camera injection | Native-vs-drop comparisons and matched-target ablations |
| Full adaptation of a two-expert 14B-class backbone is expensive | Adapt the domain on one A800 80 GB device | Rank-16 LoRA, BF16, gradient checkpointing, expert offload | Trainable-parameter, memory, throughput, and stability logs |
| Scene labels may cease to describe a moving crop | Prevent semantic label drift during benchmark construction | CLIP-based post-crop semantic gate used only for data quality control | Frozen real-CLIP audit plus human spot-check agreement |

## Core contributions for the draft

1. **A video-world-model task for remote sensing.** We formulate continuous,
   low-overlap spatial extrapolation from one observed remote-sensing view and a
   camera trajectory, separating spatial rollout from temporal change
   forecasting.
2. **A camera-aware parameter-efficient adaptation recipe.** We adapt the
   LingBot-World/Wan I2V backbone with rank-16 LoRA and explicitly compare
   camera-enabled and pose-dropped configurations.
3. **A controlled multi-direction benchmark protocol.** We construct
   source-traceable rollouts with continuous multi-segment paths, strict
   start/end overlap, aligned 17/25/33-frame horizons, and test-only transfer to
   WHU-RS19 and RESISC45.
4. **A transparent evaluation plan.** We separate frame fidelity, perceptual
   similarity, video-distribution proxies, trajectory adherence, efficiency,
   and cross-domain generalization, with no result asserted before its artifact
   exists.

## Claim-to-evidence map

| Intended claim | Code/document evidence | Current state | Allowed wording |
|---|---|---|---|
| The backbone is an I2V video diffusion transformer | `wan/modules/model.py`, `wan/configs/wan_i2v_A14B.py`, `wan/modules/vae2_1.py` | [IMPLEMENTED]; [VERIFIED] static audit and CPU import/compile reports | “The implementation uses…” |
| Video tokens use 3D positional structure | `rope_apply` splits frequency dimensions over F/H/W; Conv3D patch embedding | [IMPLEMENTED] | “joint FHW attention with 3D RoPE,” not “separate spatial and temporal modules” |
| Camera conditioning uses Pluecker rays | `wan/utils/cam_utils.py`, `new/inference.py`, `WanAttentionBlock` | [IMPLEMENTED]; [VERIFIED] pose-timeline CPU regression | “camera-coordinate conditioning,” not “explicit 3D reconstruction” |
| Canonical rollouts have T in {17,25,33} | builder uses K=2..4, 9 samples/segment, boundary deduplication | [IMPLEMENTED]; [VERIFIED] protocol tests and WHU synthetic-chain test | Exact statement is allowed |
| Canonical start/end overlap is <0.1 | builder rejects samples outside [0,0.1) and performs a final audit assertion | [IMPLEMENTED]; [VERIFIED] protocol-unit scope | State the definition; final dataset-wide artifact remains [TODO] |
| LoRA arms have the requested semantics | three YAML configs and `get_loraconfig` target filters | [IMPLEMENTED]; [VERIFIED] runner/config CPU tests reported | Exact configuration statement is allowed |
| AID is train; WHU/RESISC are test-only transfer sets | builder tiers, configs, generalization scripts/readmes | [IMPLEMENTED] | Dataset roles are allowed; performance is [TODO] |
| Evaluation includes FID, LPIPS, FVD proxy | `lora_remotesense/new/metrics.py` | [IMPLEMENTED]; known-answer tests partially verified | Always write `FVD-proxy (r3d_18)`; standard I3D-FVD remains [TODO] |
| The method improves quality/control | No current official result artifact for the three requested arms | [TODO] | Only a hypothesis or result placeholder |
| The method generalizes to WHU/RESISC | No completed AID-trained cross-domain result table | [TODO] | Only an experiment plan |
| The method beats external methods | External adapters are incomplete/not run | [TODO] | Never claim superiority |

## Important confound in the current three-arm design

The current comparison changes two factors at once:

- `native_attn_cam_r16`: native pose path + attention/camera LoRA targets;
- `drop_attn_only_r16`: pose path removed + attention-only LoRA targets.

Therefore, a difference between these arms cannot by itself identify the causal
effect of camera conditioning. **[TODO]** Add matched controls such as
`native + attn_only` and `drop + attn_cam` (with the inactive camera path
reported explicitly), or soften the claim to a comparison of complete system
configurations. The existing `all_r16_ablation` is also pose-dropped; although
camera linear layers are targetable under `all`, they are not exercised when
the camera path is absent. It is not a camera-aware upper bound.

The YAML defaults also use different evaluation caps (`400` for native and
`100` for drop/all). These caps are acceptable for staged engineering runs but
not for a headline comparison. **[TODO]** Score all methods on the identical
locked hidden-test sample IDs and report the common sample count.

## Minimum publishable evidence gate

The paper should not move from draft to submission language until all of the
following exist:

1. Dataset manifests and overlap audits for the exact official AID, WHU-RS19,
   and RESISC45 artifacts.
2. At least three seeds for the main native and drop configurations, with fixed
   split and generation seeds documented separately.
3. Real trainable-parameter audits and A800 peak-memory/runtime logs.
4. Main quality/control results with uncertainty and failure-case examples.
5. At least one credible external generative baseline under the same input,
   horizon, resolution, split, and metric implementation.
6. Test-only WHU-RS19 and RESISC45 evaluation using AID-trained weights.
7. A standard I3D-FVD result if cross-paper FVD comparison is made; otherwise
   retain the explicit `r3d_18 proxy` label.
