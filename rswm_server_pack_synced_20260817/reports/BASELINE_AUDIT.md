# RS-Roll Baseline Audit

Audit date: 2026-08-16  
Audited code baseline: `rsroll-bench-iclr-20260815-a800-whurs19-experiment-runner-safety-patched/rswm_server_pack`  
Audit scope: local source, registry wiring, model/checkpoint artifacts, recorded run products, manifest protocol, and evaluator semantics.

## Executive finding

The repository already contains a non-trivial baseline adapter layer under
`server_run/baselines/`; it is not an empty placeholder.  However, an adapter or
registry entry is not evidence that a baseline has been reproduced.  The following
facts were verified directly:

- `server_run/baselines/baseline_status.json` has `tested=false` and
  `result_available=false` for every entry.
- No `.ckpt`, `.pt`, `.pth`, `.safetensors`, or `.bin` model artifact is stored in
  this source tree.
- No `scores.json`, `aggregate_metrics.json`, or `eval_results.json` baseline result
  is stored in this source tree.
- The existing baseline runner produces an internal record with arrays shaped
  `(T,H,W,3)` `uint8`; this differs from the paper-facing result directory requested
  for this extension.
- The existing baseline runner computes mean SSIM/PSNR over `t=0...T-1`.  It does
  **not** currently enforce future-only visual scoring.  In contrast,
  `server_run/evaluate_rollout_protocol.py` correctly begins paired frame scoring at
  `CONDITION_FRAMES` and therefore excludes the condition frame.
- The standard I3D-FVD implementation already exists in
  `server_run/recompute_fvd_i3d.py` and explicitly separates `FVD_I3D` from the
  torchvision R3D-18 proxy.  Its real extractor/checkpoint has not been run or
  verified in this local tree.

## Existing interface and data contract

Registry: `server_run/baselines/registry.py`  
Runner: `server_run/baselines/run_baseline.py`  
External adapters: `server_run/baselines/external_stubs.py`

The live adapter contract is:

```text
sample dict
  -> BaselineAdapter.run(sample)
  -> {
       sample_id,
       generated_frames,
       source_frames,
       target_frames,
       route_metadata,
       runtime_metadata
     }
```

`generated_frames`, `source_frames`, and `target_frames` are required to be
`numpy.ndarray` values in `(T,H,W,3)` or `(S,H,W,3)` channel-last `uint8` form.
The requested paper-facing representation `[T,C,H,W]` can therefore be supported at
the boundary without changing any baseline internally; the repository's canonical
runtime representation remains `(T,H,W,C)`.

The existing shared manifest requires `h5`, `split`, `num_frames`, `resolution`, and
`gen_seed`, with optional `sample_keys` and `max_samples`.  It resolves sample keys
deterministically against `val_indices`.  It does **not** yet record all required
paper identity fields such as per-sample source dataset/split, trajectory ID/type,
condition-to-future overlap profile, endpoint overlap, and content checksums.

## Baseline-by-baseline audit

### Persistence

Baseline: `persistence`  
Code location: `server_run/baselines/registry.py` wrapping
`lora_remotesense/new/diagnostic_baselines.py`  
Registration status: registered; `kind=diagnostic`; adapter `status=implemented`  
Actually implemented: **YES**, deterministic CPU implementation  
Model/checkpoint: not required  
Runtime verified: **NO** in the incoming status file (`tested=false`)  
Result available: **NO** (`result_available=false`)  
Native output: repeated condition frame for the requested clip length  
Unified benchmark eligibility: **YES**, after future-only evaluator correction  
Fairness issue: high similarity near the condition frame can inflate a score if
`t=0` is included; trajectory metrics must also expose zero-motion failure  
Required change: run end-to-end synthetic CPU test; emit baseline card and standard
result directory; exclude condition frame from visual metrics.

### Deterministic translation / crop diagnostic

Baseline: `deterministic_translation`  
Code location: `server_run/baselines/registry.py` and
`lora_remotesense/new/diagnostic_baselines.py`  
Registration status: registered; adapter `status=implemented`  
Actually implemented: **YES**  
Model/checkpoint: not required  
Runtime verified: **NO** in the incoming status file  
Result available: **NO**  
Native output: trajectory-indexed crops from the full static scene  
Unified benchmark eligibility: **DIAGNOSTIC TRACK ONLY**  
Fairness issue: it consumes `full_scene` and benchmark-construction waypoints that a
deployable single-image model does not receive.  It is an oracle-like upper-bound
proxy, not a normal baseline.  The incoming registry documents this but does not set
the `upper_bound` flag used by `gt_oracle`; paper tables must still label it clearly.  
Required change: add an explicit diagnostic/oracle card and prevent inclusion in the
fully comparable main table.

### GT oracle

Baseline: `gt_oracle`  
Code location: `server_run/baselines/registry.py` and
`lora_remotesense/new/diagnostic_baselines.py`  
Registration status: registered; `upper_bound=True`  
Actually implemented: **YES**  
Model/checkpoint: not required  
Runtime verified: **NO** in the incoming status file  
Result available: **NO**  
Native output: ground-truth video passed through unchanged  
Unified benchmark eligibility: **METRIC CEILING ONLY**  
Fairness issue: directly uses the answer; it must never be ranked as a method  
Required change: baseline card and main-table exclusion guard.

### Random-direction translation

Baseline: `random_direction_translation`  
Code location: `server_run/baselines/registry.py` and
`lora_remotesense/new/diagnostic_baselines.py`  
Registration status: registered; adapter `status=implemented`  
Actually implemented: **YES**  
Model/checkpoint: not required  
Runtime verified: **NO** in the incoming status file  
Result available: **NO**  
Native output: sharp translated rollout with deliberately incorrect heading  
Unified benchmark eligibility: **DIAGNOSTIC TRACK ONLY**  
Fairness issue: uses construction geometry and is designed to fail direction control  
Required change: card and synthetic trajectory-metric test.

### Generic image outpainting

Baseline: `image_outpainting`  
Code location: `server_run/baselines/external_stubs.py` calling
`lora_remotesense/new/diagnostic_baselines.py::image_outpainting_baseline`  
Registration status: registered; adapter `status=implemented`  
Actually implemented: **ADAPTER/ROLLOUT YES; REAL MODEL RUN NO**  
Model/checkpoint: expects `stabilityai/stable-diffusion-2-inpainting` or
`RSWM_INPAINT_MODEL`; absent locally  
Runtime verified: **NO** (`tested=false`)  
Result available: **NO** (`result_available=false`)  
Native output: image inpainting; benchmark adaptation performs sequential
translate-and-inpaint rollout  
Unified benchmark eligibility: **YES, adapted image-output track**, after checkpoint
provenance and inference settings are frozen  
Fairness issue: continuous video is not the native output; recursive error propagation
is introduced by RS-Roll's adaptation.  It must be labeled `adapted trajectory`, not
`native trajectory conditioning`.  
Required change: baseline card, inference configuration snapshot, real GPU smoke, and
same-manifest evaluation.  No result may be reported before those steps.

### Remote Sensing-Oriented World Model / RSWISE-style model

Baseline: `rswise_recursive`  
Code location: `server_run/baselines/external_stubs.py`; the referenced function in
`lora_remotesense/new/diagnostic_baselines.py` raises `NotImplementedError`  
Registration status: registered; adapter `status=stub`  
Actually implemented: **NO**  
Model/checkpoint: **NO local code or checkpoint**  
Runtime verified: **NO**  
Result available: **NO**  
Native output: single direction-conditioned neighboring image, according to the paper  
Unified benchmark eligibility: **NOT YET**; only an explicitly labeled recursive or
per-waypoint bridge could map its native image output to a T-frame rollout  
Fairness issue: such a bridge is an RS-Roll adaptation, not a capability claimed by
the original method.  A direction-only model must not be given future GT or hidden
trajectory geometry.  
Required change: provide a strict ingest/bridge contract and baseline card, while
keeping lifecycle state `CONFIGURED`/`NOT RUN` until authors' runnable model and
checkpoint are available.

Public availability check: the paper is available as arXiv:2509.17808 and defines
direction-conditioned spatial extrapolation plus the 1,600-task RSWISE benchmark.
As of this audit, searches for the exact paper title, `RemoteBAGEL`, and `RSWISE`
found no verifiable official source repository or checkpoint link.  Therefore this
audit does not claim reproducibility.  Source checked:
https://arxiv.org/abs/2509.17808

### MVSplat360

Baseline: `mvsplat360`  
Code location: bridge in `server_run/baselines/external_stubs.py`; producer in
`lora_remotesense/evaluate_mvsplat360.py`  
Registration status: registered; adapter `status=unavailable`  
Actually implemented: **OUTPUT BRIDGE YES; REAL UPSTREAM CONTRACT NOT VERIFIED**  
Model/checkpoint: upstream repository and required checkpoints absent locally  
Runtime verified: **NO**  
Result available: **NO**  
Native output: pose-conditioned multi-view novel-view synthesis  
Unified benchmark eligibility: **SEPARATE MULTI-VIEW TRACK / CAPABILITY TABLE**  
Fairness issue: receives multiple context views and camera poses, unlike RS-Roll's
single condition image; it is not a fully comparable main-table baseline  
Required change: retain separate environment and black-frame success gate; generate a
card that records the unequal input contract; do not place it in the single-image
fully comparable table.

### Frozen pretrained video diffusion

Baseline: frozen Wan/LingBot I2V  
Code location: `lora_remotesense/new/evaluate_baseline.py`  
Registration status: **NOT registered in the incoming baseline registry**  
Actually implemented: **YES as a standalone evaluator**, loading the two pretrained
DiT experts without LoRA  
Model/checkpoint: configured externally; absent locally  
Runtime verified: **NO local result**  
Result available: **NO**  
Native output: video rollout  
Unified benchmark eligibility: **YES** after a registry bridge and same-manifest guard  
Fairness issue: pose route must be stated explicitly and trainable adaptation parameter
count must be exactly zero  
Required change: add result-ingest adapter/card; use the same sample manifest and frame
protocol as the adapted models.

### Same video backbone without camera conditioning

Baseline: Wan/LingBot I2V with camera condition disabled  
Code location: pose-drop infrastructure exists in training/evaluation code, but no
dedicated baseline adapter/card is registered  
Registration status: **NOT registered**  
Actually implemented: **CONFIGURABLE EXPERIMENT, NOT A VERIFIED BASELINE RESULT**  
Model/checkpoint: no trained no-camera checkpoint in this tree  
Runtime verified: **NO**  
Result available: **NO**  
Native output: video  
Unified benchmark eligibility: **YES**, provided training data/budget and test manifest
are matched  
Fairness issue: a no-camera comparison must hold LoRA target modules fixed; the existing
`native_attn_cam_r16` versus `drop_attn_only_r16` pair changes two factors at once  
Required change: register an ingest bridge/card and add matched pose x LoRA target
configurations.

### Current RS-Roll arms

Baselines/configurations:

- `native + attn_cam + r16`
- `drop + attn_only + r16`
- `drop + all + r16`

Code location: `configs/*.yaml` and the experiment runner  
Registration status: **not present in the incoming baseline registry**  
Actually implemented: configuration/runner path exists; training results absent  
Model/checkpoint: base checkpoint external; LoRA checkpoints absent locally  
Runtime verified: runner CPU/dry-run tests existed before this audit, but no formal
training/evaluation result exists in the source tree  
Result available: **NO**  
Unified benchmark eligibility: **YES after result-ingest registration**  
Fairness issue: current arms are not a matched single-factor ablation  
Required change: matched 2x2 pose/target configurations and a rank sweep that fixes
`native + attn_cam` while varying only rank.

## Evaluator audit

The incoming project has several evaluators rather than one paper-facing orchestration
layer:

- `server_run/baselines/run_baseline.py` reuses trajectory scoring, SSIM, and PSNR.
- `server_run/evaluate_rollout_protocol.py` implements condition-to-future overlap and
  future-only framewise SSIM/PSNR.
- `lora_remotesense/metrics.py` labels a torchvision R3D-18 feature distance as `FVD`,
  which is only a proxy and must be renamed at export time.
- `server_run/recompute_fvd_i3d.py` provides a separate standard I3D feature protocol
  and outputs `FVD_I3D`.

Required unification work:

1. Exclude `t=0` from all visual/video quality scores.
2. Export the existing R3D metric only as `FVD_R3D_PROXY`.
3. Preserve `FVD_I3D` as a separate nullable field and record extractor provenance.
4. Emit per-sample and aggregate JSON in one result schema.
5. Fail loudly on malformed manifests, empty generation, shape mismatch, unavailable
   baselines, or excessive sample failure.

## Manifest and overlap audit

The incoming baseline manifest pins H5, split, sample keys, frame count, resolution,
and generation seed.  It is insufficient for the requested benchmark release because
it lacks per-sample identity and checksums.  The rollout evaluator already defines
overlap correctly as:

```text
O_t = overlap(condition view at t=0, future view at t)
```

and separately audits adjacent-step overlap.  This is the correct distinction.
Difficulty bins in the incoming rollout evaluator are preregistered fixed anchors;
the requested named `Dense/Medium/Sparse/Extreme` bins must not be hard-coded until a
real manifest's overlap distribution has been analyzed.  The extension should export
distribution quantiles and leave proposed named thresholds in an analysis artifact.

## Audit conclusion

The appropriate implementation strategy is to extend the existing adapter/runner
layer, not rewrite the Video DiT or its camera mechanism.  Persistence can be made the
first fully CPU-smoke-tested baseline.  Image outpainting, RSWISE-style recursion,
frozen I2V, no-camera I2V, and RS-Roll checkpoints must remain `NOT RUN` or
`NOT VERIFIED` until their real dependencies and outputs exist.
