# Working Title (Candidate; Not Finalized)

## Camera-Aware Adaptation of Video Diffusion World Models for Low-Overlap Remote-Sensing Extrapolation

**Draft target:** ICLR submission  
**Project:** RS-Roll / RSWM-Bench  
**Draft status:** Method and protocol draft; official training results are not yet available.

### Evidence-status legend

- **[IMPLEMENTED]**: present in the inspected repository.
- **[VERIFIED]**: supported by a repository test/report, with the verification
  scope stated where material.
- **[TODO]**: not yet supported by the required experiment artifact.

Status labels are drafting aids and should be removed only after the corresponding
evidence has been archived for submission.

# Abstract

Remote-sensing world models should reason beyond an observed footprint while
preserving spatial continuity and obeying a requested viewing path. Existing
remote-sensing generation formulations commonly produce one image or an
adjacent tile from a discrete direction, leaving continuous, low-overlap spatial
rollouts under-specified and under-evaluated [REF1, REF15]. We study a different
setting: given a single remote-sensing image and a continuous camera trajectory,
generate an ordered video of spatially extrapolated views whose terminal crop
shares less than 10% area with the conditioning crop. **[IMPLEMENTED]** RS-Roll
adapts a camera-enabled LingBot-World/Wan image-to-video diffusion backbone
[REF2, REF3] using rank-16 LoRA [REF6]. Camera-to-world translations and
intrinsics are converted to Pluecker ray embeddings and injected into the video
transformer, while a CLIP-based gate [REF7] is used only to filter semantically
invalid benchmark crops. **[IMPLEMENTED]** The accompanying protocol constructs
continuous two- to four-segment paths with 17, 25, or 33 frames, enforces
source-disjoint splits, and defines AID as the adaptation source with WHU-RS19
and RESISC45 as test-only transfer domains [REF8--REF10]. We plan to evaluate
frame fidelity, perceptual similarity, video-distribution quality, trajectory
adherence, and parameter efficiency. **[TODO] Experiments are ongoing; all
method comparisons and numerical results are to be filled from official run
artifacts.**

# 1 Introduction

Remote sensing offers broad, repeated observations of the Earth, but a single
image still exposes only a bounded spatial footprint. A useful remote-sensing
world model should therefore do more than classify or caption the observed
region: it should represent how observations change as the viewing window moves
through nearby, partly or wholly unobserved space. Recent remote-sensing world
model work has begun to formulate spatial extrapolation and future-scene
generation as generative problems [REF1, REF15]. These efforts establish the
importance of generative spatial reasoning, yet also expose an open question:
what should be predicted, controlled, and measured when the desired output is a
continuous traversal rather than one synthesized image?

We focus on four gaps in current problem formulations. First, an image-to-image
adjacent-tile task does not directly measure continuity across a rollout.
Second, a small set of directional words cannot represent arbitrary headings,
turns, or path lengths. Third, high-overlap or one-step evaluation can reward
copying local appearance without testing what happens when the endpoint is
mostly unobserved. Fourth, visual plausibility alone does not establish that a
generation follows its commanded trajectory or remains spatially coherent.
These are evaluation gaps rather than a claim that every prior model has the
same architecture or failure mode. In particular, the Remote Sensing-Oriented
World Model and its RSWISE benchmark formulate direction-conditioned adjacent
image extrapolation [REF1], whereas our target is an ordered, multi-direction
video rollout under a strict low-overlap endpoint constraint.

Video diffusion models are a promising starting point because their pretraining
requires coherent evolution over an ordered sequence, and several works show
that video priors can be adapted for image-to-video generation, camera motion,
or sparse-view synthesis [REF3--REF5]. This motivation must be stated carefully:
temporal coherence is not equivalent to explicit 3D geometry, and a video model
does not become a geometrically correct world model merely by generating smooth
frames. Our hypothesis is narrower: exposing a pretrained video diffusion model
to trajectory-derived camera coordinates during parameter-efficient adaptation
should improve trajectory-conditioned remote-sensing rollouts relative to a
pose-dropped configuration. **[TODO]** This hypothesis requires controlled
experiments and is not established by implementation alone.

We introduce RS-Roll, a method-and-benchmark framework for low-overlap continuous
spatial extrapolation. **[IMPLEMENTED]** The method adapts the camera-enabled
LingBot-World/Wan I2V backbone using rank-16 LoRA and compares three predefined
system configurations: native camera conditioning with attention-and-camera
targets, pose-dropped attention-only adaptation, and a pose-dropped all-linear
ablation. **[IMPLEMENTED]** The benchmark samples continuous-angle,
multi-segment crop trajectories from remote-sensing scenes, converts them to
nadir planar camera poses and Pluecker rays, enforces a start/end overlap below
0.1, and produces aligned 17/25/33-frame clips. The design follows a diagnostic
benchmark narrative: define the evaluation gap, construct controlled tasks,
separate complementary measurements, and report failure modes rather than
compressing all behavior into one score [REF14].

## Contributions

1. **[IMPLEMENTED]** We formulate **low-overlap continuous spatial
   extrapolation** as a remote-sensing video world modeling problem: one observed
   image and a camera trajectory condition an ordered rollout of unseen space.
2. **[IMPLEMENTED]** We provide a **camera-aware LoRA adaptation recipe** for a
   LingBot-World/Wan I2V backbone, including explicit native, pose-dropped, and
   broad-target rank-16 configurations.
3. **[IMPLEMENTED]** We introduce a **multi-direction benchmark protocol** with
   continuous trajectory angles, two to four segments, 17/25/33-frame horizons,
   source-disjoint splitting, and a hard start/end overlap interval
   \(0\leq\Omega<0.1\).
4. **[IMPLEMENTED] [TODO: empirical population]** We define a transparent
   evaluation suite covering frame/perceptual fidelity, an explicitly labeled
   video-distance proxy, trajectory adherence, efficiency, and AID-to-WHU/RESISC
   transfer without reporting any unrun comparison as a result.

# 2 Related Work

## 2.1 Remote-sensing generation and world models

Remote-sensing generation spans synthesis, inpainting, change forecasting, and
spatial extrapolation. The Remote Sensing-Oriented World Model formulates
direction-conditioned spatial extrapolation from a central observation and
introduces RSWISE for evaluating generated adjacent image tiles [REF1]. More
recent remote-sensing world model formulations also combine understanding with
future-scene forecasting [REF15]. These directions motivate generative world
modeling for Earth observation, but they are not identical to our task. RS-Roll
does not forecast physical land-cover change from historical observations. It
turns a static remote-sensing scene into a controlled spatial traversal and asks
whether an I2V world model can maintain fidelity and command adherence as the
view moves into mostly unobserved space.

The source corpora in this draft—AID, WHU-RS19, and NWPU-RESISC45—were introduced
for remote-sensing scene recognition rather than video generation [REF8--REF10].
We therefore treat the resulting crop sequences as a controlled spatial
benchmark, not as real temporal satellite videos. This distinction limits the
claim but enables exact trajectories and pixel-aligned targets.

## 2.2 Video diffusion models and video world models

Latent video diffusion models extend image generation to ordered clips and can
be adapted to image-to-video and camera-motion tasks [REF4]. Wan provides an
open diffusion-transformer video foundation model family, including I2V
generation and a video VAE [REF3]. LingBot-World builds an open video-generation
world simulator and supplies the camera-enabled checkpoint used in this project
[REF2]. **[IMPLEMENTED]** Our repository retains the Wan I2V structure, including
3D patch embedding, joint attention over flattened spatiotemporal tokens, 3D
RoPE over frame/height/width axes, a temporally compressive video VAE, and
low-/high-noise diffusion experts.

Video priors have also been used in novel-view synthesis. MVSplat360 combines a
geometry-aware 3D representation with a pretrained video diffusion model to
address sparse-view, low-overlap rendering [REF5]. That result motivates video
priors for wide-baseline synthesis, but the method and evidence are different:
RS-Roll currently has no 3D Gaussian representation, depth supervision, or
multi-view input. We consequently describe our approach as **camera-coordinate
conditioned**, not as an explicit 3D reconstruction method.

## 2.3 Spatial extrapolation and diagnostic evaluation

Spatial extrapolation becomes ill-posed as overlap decreases because fewer
observed pixels constrain the target. An endpoint-only image metric is also
insufficient for a path-conditioned rollout: a model may look plausible while
moving in the wrong direction, delay a turn, or accumulate drift. Our benchmark
therefore records the full waypoint path, segment angles, per-frame crop centers,
start/end overlap, and adjacent-frame overlap audits. This supports separate
analysis of endpoint novelty and local continuity. The diagnostic philosophy is
aligned with recent world-model benchmarks that decompose visual quality,
geometry, physics, or interaction rather than treating one perceptual score as
complete [REF14]. We borrow only this organizational principle, not any task,
metric value, or experimental conclusion from that work.

## 2.4 Parameter-efficient adaptation

LoRA freezes pretrained weights and learns low-rank updates to selected linear
layers [REF6]. This is useful for the two-expert, 14B-class video backbone in our
setting, where full fine-tuning would be expensive on a single 80 GB GPU.
**[IMPLEMENTED]** RS-Roll attaches LoRA to both low- and high-noise experts and
provides target filters for attention projections, camera-injection linears,
and feed-forward linears. Our contribution is the remote-sensing adaptation and
comparison protocol, not a modification of the LoRA algorithm itself.

# 3 Problem Formulation

## 3.1 Trajectory-conditioned spatial rollout

Let \(S:\mathcal{D}\rightarrow\mathbb{R}^3\) denote a static RGB
remote-sensing scene over image domain \(\mathcal{D}\). A crop operator
\(C(S,p_t)\) extracts a window centered at 2D position
\(p_t=(c^y_t,c^x_t)\). The benchmark provides an ordered path

\[
\tau = (p_0,p_1,\ldots,p_{T-1}),
\]

obtained by linearly interpolating a sequence of waypoints. Its ground-truth
rollout is \(x_t=C(S,p_t)\). At inference, the model observes only
\(x_0\), a scene prompt \(c\), camera intrinsics \(K_t\), and camera-to-world
poses \(G_t\) derived from \(\tau\). It predicts

\[
p_\theta(\hat{x}_{1:T-1}\mid x_0,c,\{G_t,K_t\}_{t=0}^{T-1}).
\]

The stored/generated clip follows the I2V convention of \(T\) frames including
the conditioning frame. **[IMPLEMENTED]** The strict evaluation contract removes
that condition frame before fidelity and distributional scoring so that copying
the input cannot inflate future-frame results.

## 3.2 Continuous, multi-segment trajectories

**[IMPLEMENTED] [VERIFIED: protocol-unit scope]** The canonical builder samples
\(K\in\{2,3,4\}\) path segments. Each segment has nine temporal samples; the
shared boundary between consecutive segments is emitted once. Hence

\[
T=9+(K-1)(9-1)\in\{17,25,33\}.
\]

Path headings are continuous in \([0,360)\) in the canonical setting rather
than quantized to four directions. Start positions are randomized within valid
scene bounds, and later waypoints are constrained to remain crop-valid. A
separate four-direction/high-overlap profile exists as a learnability diagnostic
and must not be mixed with canonical results.

## 3.3 Low-overlap definition

Let \(B_t\subset\mathcal{D}\) be the source-scene footprint of the crop at
rollout step \(t\). We define endpoint conditioning overlap as

\[
\Omega_{0,T-1}=
\frac{|B_0\cap B_{T-1}|}{|B_0|}.
\]

The canonical acceptance rule is

\[
0\leq\Omega_{0,T-1}<0.1.
\]

**[IMPLEMENTED]** Candidates outside this interval are rejected before being
written, and a final artifact-level audit asserts the constraint. This is a
start-to-end definition. Adjacent frames intentionally retain greater overlap
to define a continuous path, and their overlap is recorded separately. We do
not claim that every neighboring pair has less than 10% overlap.

## 3.4 Coordinate and claim conventions

Waypoints use image coordinates \((c_y,c_x)\), with east/right corresponding to
0 degrees, north/up to 90 degrees, west/left to 180 degrees, and south/down to
270 degrees. The current pose generator models a fixed-orientation, nadir-looking
camera translating parallel to a plane. This is a controlled 2D-to-camera
mapping, not a complete orbital or six-degree-of-freedom satellite trajectory.

# 4 Method

## 4.1 Video diffusion backbone

**[IMPLEMENTED]** RS-Roll uses the camera-enabled LingBot-World checkpoint with a
Wan I2V diffusion-transformer backbone [REF2, REF3]. The implementation contains
two denoising experts for low- and high-noise regimes, an I2V conditioning path,
a video VAE with temporal/spatial stride \((4,8,8)\), and transformer blocks
with self-attention, text cross-attention, and feed-forward layers. A Conv3D
patch embedder converts latent video volumes into tokens. Self-attention operates
jointly on the flattened \(F\times H\times W\) token sequence; rotary frequencies
are split across frame, height, and width. We therefore call this **joint
spatiotemporal attention with 3D RoPE**, rather than claiming separately
factorized spatial and temporal attention modules.

For camera control, the path poses are interpolated to the latent frame count
and converted to relative poses. Given a ray origin \(o\) and normalized world
direction \(d\), each pixel receives the Pluecker coordinate

\[
\pi=(d,\;o\times d)\in\mathbb{R}^{6}.
\]

The ray map is patch-embedded and transformed into per-token scale and shift
terms inside every diffusion-transformer block. **[VERIFIED: CPU pose-timeline
regression]** Shape, timeline, and route contracts have automated tests;
**[TODO]** end-to-end geometric behavior on official model outputs remains an
experimental question.

## 4.2 Camera-aware LoRA adaptation

For a selected frozen linear map \(W\), LoRA learns

\[
W'=W+\frac{\alpha}{r}BA,
\]

where \(r=16\) and \(\alpha=16\) in all three requested configurations. LoRA is
attached to both diffusion experts. **[IMPLEMENTED]** The target selector maps:

- `attn_only` to self- and cross-attention projections;
- `attn_cam` to the attention projections plus camera injector, scale, and shift
  linears, while excluding feed-forward linears;
- `all` to all linear layers inside transformer blocks, including attention,
  camera, and feed-forward linears.

The three current experiment configurations are:

| Configuration | Pose path | LoRA targets | Rank | Intended role |
|---|---|---:|---:|---|
| `native_attn_cam_r16` | native Pluecker condition | `attn_cam` | 16 | main camera-aware system |
| `drop_attn_only_r16` | removed (`dit_cond_dict=None`) | `attn_only` | 16 | pose-dropped comparison |
| `all_r16_ablation` | removed | `all` | 16 | broad-target ablation |

**[VERIFIED: config/runner CPU tests reported]** The runner derives pose mode
from the YAML rather than an experiment-name heuristic. **[TODO]** The first two
systems change pose routing and LoRA target scope simultaneously. They compare
complete configurations but do not isolate the causal effect of camera input.
A matched-target crossed ablation is required before making a component-level
causal claim.

## 4.3 Semantic consistency gate

Scene-class labels describe a full source image, but a moving crop can leave the
object or terrain that justified the label. **[IMPLEMENTED]** During benchmark
construction, a CLIP-based zero-shot gate [REF7] samples frames across a crop
sequence and checks whether the declared `water`, `urban`, or `rural` semantic
pool remains supported; the mixed `general` category is intentionally skipped.
Rejected crops are resampled, and gate metadata is stored for audit.

The semantic gate is **data quality control**, not a training-loss innovation
and not part of model inference. Its prompt bank, threshold, model revision,
preprocessing, and acceptance distribution must be frozen and reported.
**[VERIFIED: injected-score CPU regression] [TODO: official real-CLIP artifact
audit]** The repository reports extensive stub/contract tests, but the paper
must not claim production-gate validity until the official dataset build records
the real local CLIP checkpoint and audit outputs.

## 4.4 Benchmark construction

**[IMPLEMENTED]** Each source image is normalized to a scene canvas, a random
valid multi-segment path is sampled, and \(128\times222\) RGB windows are
extracted and resized to \(480\times832\). Path boundaries are deduplicated to
preserve the exact 17/25/33-frame VAE contract. The builder stores frames,
waypoints, segment angles, per-frame centers, camera poses, intrinsics, source
identity, semantic-gate fields, endpoint overlap, and adjacent-step overlap.

The official AID split is grouped by source identity before assignment to
train/validation/test partitions, preventing crops from one source image from
crossing splits. **[IMPLEMENTED]** The split generator targets 70/10/20 and
supports a hidden test export with source-identifying metadata removed. WHU-RS19
and RESISC45 use separate official tiers with fail-closed source checks; they are
reserved for test-only transfer of AID-trained adapters in the current plan.

# 5 Experiments

## 5.1 Experimental setup

### Datasets

- **AID [REF8].** **[IMPLEMENTED]** Canonical adaptation source. Official runs
  will use source-disjoint train/validation/test partitions from the generated
  rollout HDF5. **[TODO]** Archive the exact 4,000-sample official train artifact,
  manifest, checksums, and overlap/semantic audits.
- **WHU-RS19 [REF9].** **[IMPLEMENTED] [VERIFIED: loader/unit and synthetic-chain
  scope]** Test-only transfer dataset. The loader handles byte/path-backed HF
  records and heterogeneous source sizes without substituting another dataset.
  **[TODO]** Archive the server-side full real-data integrity report and final
  cross-domain evaluation.
- **NWPU-RESISC45 [REF10].** **[IMPLEMENTED] [VERIFIED: checker self-test scope]**
  Test-only transfer dataset loaded from `blanchon/RESISC45` with no fallback.
  **[TODO]** Archive the official source/HDF5 checks and final cross-domain
  evaluation. The paper must report that the benchmark taxonomy uses a mapped
  subset of the 45 source classes rather than implying uniform use of every
  class.

### Optimization and hardware

The planned hardware is one NVIDIA A800-SXM4-80GB. Configurations use BF16,
micro-batch 1, gradient accumulation 4, gradient checkpointing, CPU offload of
the inactive diffusion expert, and standard AdamW. **[IMPLEMENTED] [VERIFIED:
CPU profile/preflight regression] [TODO: official end-to-end A800 training]**
The repository's local CPU checks validate configuration and failure behavior;
they do not substitute for model loading, backward, optimizer-step, peak-memory,
or convergence evidence on the target GPU.

### Evaluation metrics

We separate complementary behaviors:

1. **Frame fidelity:** FID [REF11] over generated/target future frames.
   The current implementation uses a repository-specific `torchvision`
   InceptionV3 feature path; external methods must be recomputed with the same
   code, and cross-paper comparability is not assumed.
2. **Perceptual distance:** framewise LPIPS [REF12], averaged over the rollout.
3. **Video distribution:** **FVD-proxy** using a Kinetics-pretrained
   `torchvision` R3D-18 feature extractor. **[IMPLEMENTED]** This feature space is
   not standard I3D-FVD [REF13], so proxy numbers are only comparable among runs
   evaluated by this repository. **[TODO]** Compute standard I3D-FVD before any
   cross-paper FVD comparison.
4. **Control diagnostics:** net and per-segment angular error, turn behavior,
   waypoint/path errors, and validity rates. **[IMPLEMENTED]** These metrics are
   secondary diagnostics until their official calibration artifacts are frozen.
5. **Efficiency:** trainable parameters, peak GPU memory, training/inference
   time, and throughput. **[TODO]** Populate from run manifests and logs.

All compared methods must use the same condition-frame removal, frame count,
resolution, split, generation seed policy, and metric implementation. Main
native/drop comparisons should report at least three training seeds and paired
uncertainty on the same test samples. The current YAML defaults cap native
evaluation at 400 samples and drop/all at 100; official comparison must replace
these staged caps with one locked set of identical sample IDs.

## 5.2 Main comparison

**[TODO: no official result values are available for this draft.]** Lower is
better for all three displayed metrics. `FVD-proxy` must remain visibly labeled.

| Method | Camera condition | Trainable scope | FID ↓ | LPIPS ↓ | FVD-proxy (R3D-18) ↓ |
|---|---|---|---:|---:|---:|
| Persistence diagnostic | none | none | — | — | — |
| Deterministic-translation diagnostic | commanded 2D shift | none | — | — | — |
| External image-extrapolation baseline [TODO: select and reproduce] | method-specific | method-specific | — | — | — |
| External video/I2V baseline [TODO: select and reproduce] | method-specific | method-specific | — | — | — |
| RS-Roll `drop_attn_only_r16` | no | attention LoRA | — | — | — |
| RS-Roll `native_attn_cam_r16` | Pluecker | attention + camera LoRA | — | — | — |

**Result paragraph placeholder.** `[RESULT TO BE FILLED FROM THE LOCKED
MULTI-SEED ARTIFACTS. Report effect sizes and uncertainty; do not write
“outperforms” unless all protocol checks and matched comparisons pass.]`

## 5.3 Ablations and diagnostic questions

### Requested three-arm comparison

| Arm | Pose mode | LoRA target | Rank | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Trajectory metric(s) |
|---|---|---|---:|---:|---:|---:|---:|
| Native | native | `attn_cam` | 16 | — | — | — | — |
| Drop | drop | `attn_only` | 16 | — | — | — | — |
| All | drop | `all` | 16 | — | — | — | — |

### Additional ablations required for defensible attribution

- **[TODO] Matched pose ablation:** native versus drop with the same attention
  target scope.
- **[TODO] Matched target ablation:** `attn_only` versus `attn_cam` with the same
  pose route.
- **[TODO] Rank scaling:** at minimum a small set around rank 16, reporting
  trainable parameters and resource cost rather than quality alone.
- **[TODO] Horizon scaling:** 17 versus 25 versus 33 frames, with per-horizon
  sample counts and uncertainty.
- **[TODO] Overlap analysis:** canonical endpoint overlap strata below 0.1 plus
  the separately preregistered high-overlap learnability control; never merge
  the two profiles into one headline number.
- **[TODO] Semantic-gate audit:** on/off or accepted/rejected subset analysis for
  dataset bias, without presenting the gate as a model component.

## 5.4 Cross-dataset generalization

Adapters must be trained only on the locked AID training split, then evaluated
without further tuning on WHU-RS19 and RESISC45.

| Training source | Test source | Method | FID ↓ | LPIPS ↓ | FVD-proxy ↓ | Control score | Status |
|---|---|---|---:|---:|---:|---:|---|
| AID | AID hidden test | Drop r16 | — | — | — | — | [TODO] |
| AID | AID hidden test | Native r16 | — | — | — | — | [TODO] |
| AID | WHU-RS19 | Drop r16 | — | — | — | — | [TODO] |
| AID | WHU-RS19 | Native r16 | — | — | — | — | [TODO] |
| AID | RESISC45 | Drop r16 | — | — | — | — | [TODO] |
| AID | RESISC45 | Native r16 | — | — | — | — | [TODO] |

The analysis should report both absolute target-domain performance and the
paired change relative to AID. A failure to transfer is still informative and
must be reported rather than filtered from the table.

## 5.5 Qualitative and failure analysis

**[TODO]** For fixed test trajectories and generation seeds, show the condition
frame, ground-truth rollout, each main system, and at least one external
baseline. Include examples of:

- correct direction but semantic drift;
- visually plausible frames with incorrect turns;
- long-horizon duplication or texture collapse;
- failure in water, dense urban, rural, and mixed scenes;
- a low-overlap success and a low-overlap failure at each horizon.

Examples must be selected by a preregistered or transparent rule (for example,
median/best/worst by a specified metric), not only by visual appeal.

# 6 Limitations

**Static-scene simulation gap.** The benchmark creates ordered crops from static
scene-classification imagery. It measures spatial traversal under known crop
geometry, not temporal forecasting of land-cover change, weather, illumination,
or sensor dynamics. Real multi-temporal and along-track satellite video are
needed before extending claims to Earth dynamics.

**Restricted camera model.** Current trajectories use identity rotation,
nadir-looking planar translation, zero vertical translation, and heuristic
normalization by crop dimensions. They omit attitude changes, altitude, terrain
relief, sensor projection variation, rolling shutter, and orbital mechanics.

**No explicit 3D representation.** Pluecker conditioning supplies ray
coordinates, but the model has no depth supervision or reconstructed 3D scene.
Smooth video does not prove 3D consistency. A geometry-aware evaluator or
reconstruction-based audit is required for stronger claims.

**Endpoint overlap is not adjacent overlap.** The `<0.1` constraint compares
the first and last crop. Neighboring frames overlap substantially by design.
The task is therefore low-overlap endpoint extrapolation through a continuous
bridge, not independent generation between every pair of disjoint views.

**Dataset scale and taxonomy.** AID, WHU-RS19, and RESISC45 are scene-label
datasets with heterogeneous collection processes and no trajectory ground
truth. The semantic pool mapping covers only selected classes and can introduce
category imbalance. Official artifact sizes, accepted/rejected counts, and
per-class results must be disclosed.

**Semantic-gate bias.** CLIP is trained outside the remote-sensing domain and
may reject difficult but valid crops or accept semantically incorrect ones.
Its prompt bank and threshold encode choices that can change the benchmark
distribution. Human audits and frozen checkpoint/preprocessing identifiers are
necessary.

**Evaluation mismatch.** FID and LPIPS are image-centric, while the current FVD
is an R3D-18 proxy. The repository FID path is also not asserted to match every
published FID implementation. None alone establishes controllability or
geometry. Same-code FID, standard I3D-FVD, trajectory diagnostics, human inspection, and possibly
reconstruction-based consistency should be reported together.

**Experimental confounding.** The current native and drop arms alter both pose
routing and LoRA target scope. The comparison is valid between complete systems
but insufficient for causal attribution. Matched controls are a prerequisite
for claiming that any measured difference is specifically caused by camera
conditioning.

**Compute and baseline coverage.** The two-expert backbone requires aggressive
offload on a single A800 80 GB GPU, making multi-seed evaluation costly. External
generative baselines and full official training are not complete in this draft.

# 7 Conclusion

We present an evidence-bounded draft of RS-Roll, which reframes remote-sensing
spatial extrapolation as a camera-trajectory-conditioned video rollout problem.
The implemented system connects continuous crop paths to a LingBot-World/Wan
I2V backbone through Pluecker ray conditioning and rank-16 LoRA, while the
benchmark enforces source traceability, aligned rollout horizons, and less than
10% start/end overlap. The code and CPU protocol checks establish that the task,
conditioning route, experiment arms, and evaluation interfaces exist. They do
not establish performance. **[TODO]** The paper's empirical conclusion will be
written only after official multi-seed A800 training, external baselines,
cross-dataset evaluation, and metric audits are complete.

# Preliminary references

These entries were checked against primary paper or official project pages.
Numbering is provisional; citation formatting should be converted to the final
ICLR bibliography.

- **[REF1]** Y. Lu et al. *Remote Sensing-Oriented World Model.* arXiv:2509.17808, 2025. <https://arxiv.org/abs/2509.17808>
- **[REF2]** Robbyant Team et al. *Advancing Open-source World Models.* arXiv:2601.20540, 2026. <https://arxiv.org/abs/2601.20540>
- **[REF3]** Wan Team et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314, 2025. <https://arxiv.org/abs/2503.20314>
- **[REF4]** A. Blattmann et al. *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets.* arXiv:2311.15127, 2023. <https://arxiv.org/abs/2311.15127>
- **[REF5]** Y. Chen et al. *MVSplat360: Feed-Forward 360 Scene Synthesis from Sparse Views.* NeurIPS, 2024. <https://arxiv.org/abs/2411.04924>
- **[REF6]** E. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR, 2022. <https://arxiv.org/abs/2106.09685>
- **[REF7]** A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML, 2021. <https://arxiv.org/abs/2103.00020>
- **[REF8]** G.-S. Xia et al. *AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification.* IEEE TGRS, 2017. <https://arxiv.org/abs/1608.05167>
- **[REF9]** G.-S. Xia et al. *Structural High-Resolution Satellite Image Indexing.* ISPRS TC VII Symposium, 2010; D. Dai and W. Yang. *Satellite Image Classification via Two-Layer Sparse Coding With Biased Image Representation.* IEEE GRSL, 2011. Official dataset page: <https://captain-whu.github.io/BED4RS/>
- **[REF10]** G. Cheng, J. Han, and X. Lu. *Remote Sensing Image Scene Classification: Benchmark and State of the Art.* Proceedings of the IEEE, 2017. <https://arxiv.org/abs/1703.00121>
- **[REF11]** M. Heusel et al. *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium.* NeurIPS, 2017. <https://arxiv.org/abs/1706.08500>
- **[REF12]** R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR, 2018. <https://arxiv.org/abs/1801.03924>
- **[REF13]** T. Unterthiner et al. *Towards Accurate Generative Models of Video: A New Metric & Challenges.* arXiv:1812.01717, 2018. <https://arxiv.org/abs/1812.01717>
- **[REF14]** Y. Zhao et al. *WorldOlympiad: Can Your World Model Survive a Triathlon?* arXiv:2606.11129, 2026. <https://arxiv.org/abs/2606.11129>
- **[REF15]** L. Xu et al. *RS-WorldModel: a Unified Model for Remote Sensing Understanding and Future Sense Forecasting.* arXiv:2603.14941, 2026. <https://arxiv.org/abs/2603.14941>
