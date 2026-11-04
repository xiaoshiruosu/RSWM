# RSWM Baselines — Contract, Classes, and Honest Status

> **Deployment hardware status:** primary validated target is
> **NVIDIA A800-SXM4-80GB** (sm80, BF16, FA2→SDPA). H200 141GB remains a
> legacy/reference path (sm90/optional FA3) and was not revalidated this round.

Documentation for the **already-existing** baseline adapter layer in `server_run/baselines/`.
This file documents code; it does not add any.

> ## Read this first
>
> **No baseline has ever been run. Not one number exists.**
> Every entry in `baseline_status.json` is `tested=false` **and** `result_available=false`.
> **The paper MUST NOT claim a completed external-baseline comparison.** There is no
> external generative baseline result to report. [updated v2.6] Of the three external
> candidates: `image_outpainting` is **implemented** (v2.5, SD-2-inpainting rollout,
> never yet run on the server); `mvsplat360` has a **written ingest bridge** but remains
> `unavailable` (repo/weights/env absent, model-call contract unverified — smoke gate
> first); `rswise_recursive` is still a pure stub raising `NotImplementedError`.

### What was actually executed to write this doc

The authoring sandbox has **no numpy / cv2 / h5py / torch and no GPU**, so only stdlib-level
checks were possible. Executed, and passed:

| Command | Result |
|---|---|
| `python3 -m py_compile registry.py external_stubs.py run_baseline.py __init__.py` | exit 0 |
| `python3 server_run/baselines/registry.py` (stdlib-only wiring self-check) | exit 0 — "接线自检全部通过" |
| `python3 server_run/baselines/run_baseline.py --list` | exit 0, listed 7 adapters |
| `bash -n server_run/pbs/baseline_eval_template.pbs` | exit 0 |

**Not executed** (cannot be, here): any generation, any metric, any `validate_record()`
shape/dtype branch, any H5 read, any GPU path. The smoke run printed
`numpy 不在 → 跳过形状/dtype 检查(未测,不代表通过)` — the shape/dtype half of the
contract is **unverified**, not "passing".

---

## 1. Registered adapters (verbatim `--list` output)

```
deterministic_translation      diagnostic           implemented   gpu=False
gt_oracle                      diagnostic           implemented   gpu=False  [UPPER BOUND, 非方法]
image_outpainting              external_generative  implemented   gpu=True   (v2.5)
mvsplat360                     external_generative  unavailable   gpu=True   (v2.6: bridge written)
persistence                    diagnostic           implemented   gpu=False
random_direction_translation   diagnostic           implemented   gpu=False
rswise_recursive               external_generative  stub          gpu=True
```

`status` semantics (`registry.py:76-79`):

| status | meaning |
|---|---|
| `implemented` | `run()` produces a contract-compliant record **right now**. |
| `stub` | Adapter exists, the method itself is unwritten — no runnable implementation in the repo. |
| `unavailable` | The repo **has** implementation code, but a dependency (repo/weights/env) is missing. |

> **Naming trap.** The registered name is **`random_direction_translation`**, but
> `run_baselines()` returns that video under the key **`random_direction`**
> (`diagnostic_baselines.py:436`; bridged via `output_key` at `registry.py:465-467`).
> Use the *registered* name on the CLI; the output key is internal.

---

## 2. The unified record schema

Every baseline — diagnostic or external — must return **exactly** this structure, so one
scoring path serves every row of every table. Enforced by `validate_record()`
(`registry.py:185-270`), which **raises on any deviation and never silently passes**.

Key tuples are defined at `registry.py:51-74`.

### Top level — `RECORD_KEYS` (`registry.py:51-58`)

| Key | Type / constraint | Checked at |
|---|---|---|
| `sample_id` | non-empty `str` | `registry.py:197-199` |
| `generated_frames` | `np.ndarray` `(T,H,W,3)` `uint8`, `T>=1` | `registry.py:240`, `172-182` |
| `source_frames` | `np.ndarray` `(T,H,W,3)` `uint8` | `registry.py:242` |
| `target_frames` | `np.ndarray` `(T,H,W,3)` `uint8` | `registry.py:241` |
| `route_metadata` | `dict`, exactly `ROUTE_KEYS` | `registry.py:201-202` |
| `runtime_metadata` | `dict`, exactly `RUNTIME_KEYS` | `registry.py:216-217` |

### `route_metadata` — `ROUTE_KEYS` (`registry.py:59-66`)

| Key | Type / constraint | Checked at |
|---|---|---|
| `prompt` | non-empty `str` | `registry.py:203-204` |
| `waypoints` | `[K+1, 2]`, `(cy, cx)` **camera/window-centre** pixel path, `>=2` points | `registry.py:251-255` |
| `segment_angles` | `[K]` **camera** segment headings, degrees | `registry.py:256-260` |
| `num_segments` | `int >= 1`, and must equal `K` | `registry.py:211-214`, `261-262` |
| `direction_word` | one of `("east","north","west","south")` (`registry.py:49`) | `registry.py:208-210` |
| `condition_type` | one of `("text_only","text_plucker","none")` (`registry.py:48`) | `registry.py:205-207` |

### `runtime_metadata` — `RUNTIME_KEYS` (`registry.py:67-74`)

| Key | Type / constraint | Checked at |
|---|---|---|
| `wall_seconds` | `float >= 0` | `registry.py:218-219` |
| `gen_seed` | `int` (bool rejected) | `registry.py:220-222` |
| `resolution` | `[H, W]`, two `int`s; must equal actual `generated_frames` `(H,W)` | `registry.py:226-229`, `264-266` |
| `num_frames` | `int`; must equal actual `generated_frames.shape[0]` | `registry.py:220-222`, `267-269` |
| `device` | non-empty `str` | `registry.py:223-225` |
| `method_version` | non-empty `str` (diagnostics embed a sha256 fingerprint of `diagnostic_baselines.py`, `registry.py:441`) | `registry.py:223-225` |

### Cross-field invariants

- `generated_frames.shape == target_frames.shape` — same resolution *and* frame count (`registry.py:244-246`).
- `source_frames.shape[1:] == generated_frames.shape[1:]` — same `(H,W,3)` (`registry.py:247-249`).
- `len(segment_angles) == len(waypoints) - 1 == num_segments` (`registry.py:256-262`).

### Two properties worth knowing

1. **Exact keys, both directions.** `_check_exact_keys` (`registry.py:160-169`) rejects
   **missing *and* extra** keys. You cannot smuggle an extra field into a record.
2. **Missing numpy is a hard error, not a skip.** `validate_record()` does all
   numpy-free checks first, then raises `RuntimeError` if numpy is absent
   (`registry.py:232-238`) rather than returning early — a half-validated record is never
   reported as valid. This is why the layer is importable here but not *usable* here.

### Convention notes (do not "fix" these — they are load-bearing)

- `waypoints` are `(cy, cx)` **camera/window-centre** coordinates.
- Camera heading `= degrees(atan2(-dy, dx)) % 360`; east=0 / north=90 / west=180 / south=270.
- **Content optical flow = camera motion negated** (window moves east → picture content
  flows west). `direction_word` records the **camera** direction word (the word in the
  builder's prompt). **Do not flip it 180° again yourself** — `registry.py:299-328`,
  `diagnostic_baselines.py:39-42`.
- `K` segments contain `K-1` turns.

---

## 3. `BaselineAdapter` interface & adding a baseline

Base class: `registry.py:86-111`.

```python
class BaselineAdapter:
    name: str = ""              # unique registration name
    kind: str = "diagnostic"    # 'diagnostic' | 'external_generative'   (KINDS, registry.py:46)
    requires_gpu: bool = False
    status: str = "stub"        # 'implemented' | 'stub' | 'unavailable' (STATUSES, registry.py:47)
    upper_bound: bool = False   # True = metric ceiling, NOT a deployable method
    description: str = ""

    def run(self, sample: dict) -> dict:   # must return validate_record(record)
        ...
```

Registry API: `register(adapter)` (`registry.py:121-138`), `get(name)` (`141-145`),
`list_baselines(kind=None)` (`148-153`).

`register()` fails loudly on: non-`BaselineAdapter` instance, empty `name`, duplicate
`name`, `kind` outside `KINDS`, `status` outside `STATUSES`, non-bool `requires_gpu`, or a
non-callable `run`.

**To add a baseline:**

1. Subclass `BaselineAdapter` (or `ExternalStub`, `external_stubs.py:31-62`, for external
   generative models).
2. Set `name` / `kind` / `status` / `requires_gpu` / `description` honestly. A method you
   have not run is **never** `implemented`.
3. Implement `run(sample) -> dict` and **return `validate_record(record)`** — do not
   hand-roll the checks.
4. `register(YourAdapter())` at module import.
5. Add a matching entry to `baseline_status.json` (§6) — the self-check reconciles the two
   and fails if they disagree (`registry.py:622-646`).
6. Run `python3 server_run/baselines/registry.py`; it must exit 0.

`sample` dict required fields (`registry.py:331-337`, assembled by
`run_baseline.load_sample`, `run_baseline.py:98-116`):
`sample_id`, `prompt`, `waypoints`, `segment_angles`, `window_size`, `gt_frames`,
`gen_seed` (+ optional `scene_image`/`full_scene`, `source_frame`, `T`/`num_frames`).

---

## 4. The two classes — and why the distinction is not cosmetic

### A. DIAGNOSTIC baselines — **not deployable methods**

`persistence`, `deterministic_translation`, `random_direction_translation`, `gt_oracle`.

These are **non-generative videos synthesized from a static scene + the GT trajectory**
(`diagnostic_baselines.py:2-45`). They read neither the prompt nor the poses — hence
`condition_type` is hard-set to `'none'` (`registry.py:433`). Their entire job is to prove
the benchmark **discriminates failure modes** instead of rewarding one degenerate strategy:

| Baseline | Failure mode probed | Expected signature |
|---|---|---|
| `persistence` | **near-static** — "the model just holds the input image" | ~0 flow magnitude, direction unestimable, SSIM high only on earliest frames |
| `random_direction_translation` | **good-visual / wrong-direction** | sharp frames, *correct* flow magnitude, large angular error |
| `deterministic_translation` | **right-direction / quality ceiling** | near-perfect direction & path, high SSIM |
| `gt_oracle` | **metric ceiling** | SSIM=1, PSNR=inf, angular error at the quantization floor |

> ### These two are UPPER BOUNDS that see data a real model never gets
>
> - **`gt_oracle`** passes the **GT frames through verbatim** — it has literally seen the
>   answer. Flagged `upper_bound=True` (`registry.py:478`). The self-check asserts the
>   `upper_bound` flag is set on **exactly** `{gt_oracle}` (`registry.py:555-557`).
> - **`deterministic_translation`** is a *construction-aware* upper bound: it crops the true
>   window along the GT waypoints, which requires **the full static scene** — an input a
>   real world model never receives. Strong direction/path numbers here are a **scene
>   lookup, not generation**. It is deliberately **not** given the `upper_bound` flag
>   (that flag is reserved for "read the answer directly"); its bound status lives in its
>   `description` (`registry.py:459-462`) and its `baseline_status.json` note.
>
> **Neither may ever be presented as an external method comparison, or ranked against real
> methods.** Report them as upper-bound / diagnostic rows, explicitly labelled.
> `run_baseline.py` prints this warning twice per run (`run_baseline.py:159-161`, `263-264`).

Why the adapters wrap `run_baselines()` wholesale rather than calling the four synthesizers
directly (`registry.py:344-359`): `random_direction`'s `dir_index` (the GT quadrant's
opposite) and `step_px` (speed-matched to the GT path) are derived **only inside**
`run_baselines()` (`diagnostic_baselines.py:423-436`), and it also resizes all four to one
common scoring resolution (`diagnostic_baselines.py:405-411`). Re-deriving them elsewhere
would be duplicate logic with drifting numbers.

### B. EXTERNAL generative baselines — **stubs only, zero results**

Only candidates with a **real hook in this repo** are registered; inventing a model or a
number is worse than a blank (`external_stubs.py:5-12`). CamI2V, standalone SVD video
generation, and other world models were grepped for and found **nowhere** in the repo, so
they are deliberately **not** registered.

Blocking dependencies, quoted from `baseline_status.json`:

#### `mvsplat360` — `status=unavailable` (bridge written in v2.6; repo/weights/env still absent)

> [v2.6] The record-contract **bridge IS now written**: run
> `lora_remotesense/evaluate_mvsplat360.py --no_metrics` in the separate `mvsplat` env
> (torch==2.2.2), then in the scoring env
> `export RSWM_MVSPLAT_EVAL_DIR=<output_dir>` and
> `run_baseline.py --baseline mvsplat360 --skip-failed --min-success 0.8`.
> Per-sample failures / black-frame products raise `SampleUnavailable` and are counted
> explicitly — never silently averaged. Remaining blockers: repo not staged
> (`MVSPLAT=1 bash server_run/offline_prefetch.sh` → `$STAGE/mvsplat360`), three
> checkpoints absent (`svd.safetensors` is HF-gated), and the
> `ModelWrapper(model_input)` call in `run_mvsplat360_single` has **never run against the
> real repo** — `bash server_run/mvsplat_smoke.sh` (2–5 samples + black-frame gate) is
> the mandatory first-run arbiter before any full run.

v2.6 fixes worth knowing: `p6_mvsplat.sh` now subsets from **GEO_H5** (the old SEM_H5
wiring fed neutralized identity poses to a pose-conditioned NVS — a real pre-existing
bug); `evaluate_mvsplat360.py` now imports the canonical `new/evaluate.py`, seeds each
sample (`--gen_seed`), and writes `error` / `infer_seconds` / v2.5-v2.6 caliber markers
into meta. Fairness footnote obligation: MVSplat360 receives 5 context views + poses
(more conditioning than RSWM), and context slots carry GT frames by protocol.
Bridge self-test: `python3 server_run/baselines/external_stubs.py` (6 blocks, fake
products; the real-repo contract is exactly what the smoke gate decides).

#### `image_outpainting` — `status=implemented` (v2.5; SD-2-inpainting rollout)

Implemented in v2.5: `diagnostic_baselines.image_outpainting_baseline` is a real
translate+outpaint rollout (per-frame shift along the content direction, leading-edge
strip inpainted, known region force-kept; hard error without a model — never degrades
into plain translation). Weights: `BASELINE_A=1 bash server_run/offline_prefetch.sh`
(stabilityai/stable-diffusion-2-inpainting, non-gated, ~5GB; `RSWM_INPAINT_MODEL`
overrides). Direction comes from GT waypoints quantization (not prompt parsing), so it
works on direction-stripped variants. Seeding: the generator is passed explicitly into
every pipeline call (`RSWM_INPAINT_STEPS` controls steps; same seed → bit-identical,
fake-pipe verified). `tested` remains `false` until it runs on the server.

#### `rswise_recursive` — `status=stub` (the model is simply absent)

> No recursive world-model implementation or weights exist in this repo. The RSVideoWise
> asset present (`wan/remotesensebenchmark/build_rsvideowise_benchmark.py`) is a
> **BENCHMARK BUILDER, not a model**. Only a `NotImplementedError` function stub exists
> (`lora_remotesense/new/diagnostic_baselines.py:483-513`).
> — `baseline_status.json:66`

GPU-hour cost is **not** estimated, because no model has been chosen. No number is fabricated.

---

## 5. The fairness contract

Every method in a comparison table must share: same train/val split, same sample manifest,
same generation resolution, same frame count, same `gen_seed`, same evaluation scripts.
**Enforcement is uneven — know which is which before you write a fairness claim.**

| Requirement | Status | Where |
|---|---|---|
| Same H5 + same split file | **Enforced** — manifest fields are mandatory; missing file → `exit 2` | `run_baseline.py:42`, `50-64` |
| Same sample set | **Enforced** — `sorted(f['data'].keys())` indexed by `split["val_indices"]`, identical rule to `run_trajectory_eval.py:70-74`; deterministic. `manifest.sample_keys` pins it harder | `run_baseline.py:67-95` |
| Same generation resolution | **Enforced twice** — record self-consistency, then record-vs-manifest (`_fatal`, non-zero exit) | `registry.py:264-266`; `run_baseline.py:209-211` |
| Same frame count | **Enforced twice** — same pattern | `registry.py:267-269`; `run_baseline.py:212-214` |
| Same evaluation scripts | **Enforced by construction** — reuses `trajectory_metrics.score_trajectory`, `run_trajectory_eval._flatten/_agg/TRACKB_KEYS`, `metrics.compute_ssim/compute_psnr` rather than reimplementing | `run_baseline.py:185-190`, `218-229`, `237-240` |
| Same `gen_seed` | **Recorded, but only a convention for diagnostics.** It is read from the manifest, threaded into the sample, and stored in the record — but the four diagnostic baselines are **deterministic and ignore it** (`registry.py:437`: "记录仅为协议对齐"). Nothing asserts a record's `gen_seed` equals the manifest's. It becomes load-bearing only once a real stochastic external model is wired. | `run_baseline.py:154`, `115`; `registry.py:437` |
| Split matches the one RSWM trained with | **Convention only.** The manifest simply names a split file; nothing verifies it is the same split used to train the model you are comparing against. **Point the manifest at the split from `$T_MAIN`** (`common.sh:54`, `T_MAIN="$WORK/train_r16_sem_s42"`) and say so in the paper. | — |

### Known contract limitation (must be footnoted, not hidden)

`condition_type` is a three-way enum `{text_only, text_plucker, none}` (`registry.py:48`) and
**cannot express "5 context views + camera poses"**. MVSplat360 receives
`--num_context 5` views **plus** poses (`p6_mvsplat.sh 推理段(现 :103)`) — i.e. **strictly more
conditioning than RSWM's single image + text**. It would be registered as `'none'`, which
understates its advantage. So MVSplat360 is **not a weaker opponent**; the honest framing is
"given *more* information, a general-purpose NVS model still shows a domain gap."
(`external_stubs.py:76-83`.)

---

## 6. Shared manifest

`run_baseline.py:53` and `baseline_eval_template.pbs:35` both point here for the template.
Required fields — `MANIFEST_REQUIRED` (`run_baseline.py:42`); a missing one exits `2`.

```json
{
  "h5":         "/scratch/rx40/$USER/RSWM/work_baselines/data/train_semantic.h5",
  "split":      "/scratch/rx40/$USER/RSWM/work_baselines/train_r16_sem_s42/train_val_split.json",
  "num_frames": 33,
  "resolution": [480, 832],
  "gen_seed":   42,

  "sample_keys": null,
  "max_samples": null
}
```

| Field | Required | Notes |
|---|---|---|
| `h5` | yes | Must exist (`run_baseline.py:61-63`). Read-only. Needs `data/<key>/{image,video_frames,waypoints,segment_angles}` + `prompt` attr (`run_baseline.py:98-116`) **and** a `metadata` group with a `pose_window_size` attr (`run_baseline.py:196-202`) — without it, window-cropping baselines cannot run. |
| `split` | yes | Must exist; JSON containing `val_indices` (`run_baseline.py:76`). |
| `num_frames` | yes | Hard-checked against every record. |
| `resolution` | yes | `[H, W]`, exactly two entries (`run_baseline.py:58-60`). |
| `gen_seed` | yes | See the fairness caveat above. |
| `sample_keys` | optional | Explicit key list = strongest reproducibility; every key must exist in the H5 (`run_baseline.py:82-87`). |
| `max_samples` | optional | Truncates the resolved key list (`run_baseline.py:90-92`). |

**The values above are a template, not measured settings** — fill `num_frames` /
`resolution` from the H5 and the config you are actually comparing against.

---

## 7. Running a baseline

### CLI (`run_baseline.py:119-131`)

```bash
# List every registered baseline (no numpy needed, no manifest needed)
python3 server_run/baselines/run_baseline.py --list

# Dry run — checks adapter/deps/files/sample resolution; generates nothing, writes nothing
python3 server_run/baselines/run_baseline.py \
    --baseline persistence \
    --manifest "$WORK/baselines/manifest.json" \
    --out      "$WORK/baselines/persistence" \
    --dryrun

# Real run (needs numpy/h5py/cv2 — i.e. the rswm conda env)
python3 server_run/baselines/run_baseline.py \
    --baseline persistence \
    --manifest "$WORK/baselines/manifest.json" \
    --out      "$WORK/baselines/persistence"
```

Per-baseline commands — substitute `--baseline` / `--out`:

| Baseline | `--baseline` value | Outcome today |
|---|---|---|
| Persistence | `persistence` | runs (CPU) |
| Deterministic translation | `deterministic_translation` | runs (CPU) — **upper bound** |
| Wrong-direction control | `random_direction_translation` | runs (CPU) |
| GT oracle | `gt_oracle` | runs (CPU) — **upper bound**, prints a warning |
| MVSplat360 | `mvsplat360` | **`NotImplementedError`**, lists missing deps |
| Image outpainting | `image_outpainting` | **`NotImplementedError`** |
| RSWISE recursive | `rswise_recursive` | **`NotImplementedError`** |

Flags: `--save-frames` also writes a single `frames.npz`; `--dryrun` checks only; `--list` exits after listing.

### PBS (`server_run/pbs/baseline_eval_template.pbs`)

```bash
# Dry run on the login node — no queue, no GPU, no disk writes
DRYRUN=1 BL_NAME=persistence \
  MANIFEST=/scratch/rx40/$USER/RSWM/baselines/manifest.json \
  WORK=/scratch/rx40/$USER/RSWM/work_baselines \
  bash server_run/pbs/baseline_eval_template.pbs

# Submit (WORK must be an independent root)
qsub -v BL_NAME=persistence,MANIFEST=/scratch/rx40/$USER/RSWM/baselines/manifest.json,\
WORK=/scratch/rx40/$USER/RSWM/work_baselines server_run/pbs/baseline_eval_template.pbs
```

Guards, in order (`baseline_eval_template.pbs:33-81`): `BL_NAME`/`MANIFEST` mandatory →
**`WORK` must not resolve to the old AID `work` root** (refuses, exit 1 — `:45-50`) →
refuses to overwrite an existing `scores.json` unless `FORCE=1` (`:52-57`) → code + manifest
exist (`:59-62`) → manifest fields + `h5`/`split` files exist (`:64-81`). Optional
`SAVE_FRAMES=1`. Writes only to `$WORK/baselines/<BL_NAME>/`; manifest `h5`/`split` are
read-only. **The old AID results under `server_run/work/` are never touched.**

### Outputs — deliberately few, aggregated files

`qk73` is space-tight and `rx40` is inode-tight, so each run writes **2-3 files total**, not
thousands (`run_baseline.py:20-23`, `254-259`):

| File | Contents |
|---|---|
| `<out>/scores.json` | aggregate + per-sample Track B / SSIM / PSNR |
| `<out>/resolved_manifest.json` | the exact sample keys used + protocol params (audit trail) |
| `<out>/frames.npz` | only with `--save-frames`; **all** generated frames in one compressed archive |

---

## 8. Maintaining `baseline_status.json`

Fields per entry: `name`, `kind`, `implemented`, `tested`, `result_available`,
`blocking_dependency`, `note`.

Intended meaning:

- **`implemented`** — mirrors `adapter.status == "implemented"`. The self-check fails if
  they contradict (`registry.py:639-640`).
- **`tested`** — flip to `true` **only after a real runtime run in a numpy/GPU environment
  actually produced `scores.json`**. Not after a dry run. Not after `py_compile`.
- **`result_available`** — flip to `true` **only when a number exists that you are willing
  to put in the paper**, produced under the §5 fairness contract.
- **`blocking_dependency`** — mandatory whenever `implemented` is false; the self-check
  fails on an unimplemented entry with an empty blocker (`registry.py:645-646`).

**All entries are currently `tested=false` and `result_available=false`.**

> ### Flipping a flag to `true` will FAIL the self-check — by design
>
> `registry.py:641-644` currently **asserts** `tested` and `result_available` are false for
> every entry, and `run_baseline.py:247` hard-codes `"tested": False` into every emitted
> `scores.json`. This is a tripwire, not a bug: it makes "we have results now" an explicit,
> reviewable edit to the self-check rather than a quiet JSON tweak. When a baseline genuinely
> runs, update **all three** together — the JSON entry, the assertion at `registry.py:641-644`,
> and the hard-coded `tested` at `run_baseline.py:247` — in one deliberate commit.

The self-check also reconciles the JSON's entry set against the live registry and fails on
any drift (`registry.py:622-646`). After **any** edit here, run:

```bash
python3 server_run/baselines/registry.py   # must exit 0
```

---

## 9. What to do next, to actually land an external baseline

Ordered. `mvsplat360` is the only candidate with real code in-repo, so it is the shortest path.

1. **Decide the claim first.** MVSplat360 gets 5 context views + poses — *more* conditioning
   than RSWM. Agree up front that the claim is "general NVS shows a domain gap even with more
   input", not "we beat MVSplat360". If that claim isn't worth the GPU hours, stop here.
2. **Build the separate env.** `torch==2.2.2` + `torchvision==0.17.2`, python 3.10
   (`setup_mvsplat360_autodl.sh:57`). It is **incompatible** with the main training env;
   `p6_mvsplat.sh 环境守卫段(现 :53-57)` refuses to run if `flash_attn` imports.
3. **Clone the repo + fetch weights.** `donydchen/mvsplat360` (`setup_mvsplat360_autodl.sh:42`);
   `re10k.ckpt`, `dl3dv_480p.ckpt`, and `svd.safetensors` into `$MVSPLAT_DIR/checkpoints/`.
   **`svd.safetensors` (~9.5GB) is HF-gated**: accept on the HF page, then
   `huggingface-cli login` (`setup_mvsplat360_autodl.sh:109-117`). Terms are whatever the HF
   page says — this repo does not restate them. Mind `qk73` space for ~10GB of weights.
4. **Verify deps honestly:**
   `python3 server_run/baselines/run_baseline.py --baseline mvsplat360 --manifest ... --out ... --dryrun`
   prints each missing item via `missing_dependencies()`. Iterate until it prints none.
5. **Freeze the manifest** (§6) against the **same split as the RSWM config you compare to**
   (`$T_MAIN`, `common.sh:54`) and the same resolution/frame count. Prefer explicit
   `sample_keys`. This is the step that makes the table legitimate — the split match is
   convention-only and nothing will catch you.
6. **Write the bridge.** `evaluate_mvsplat360.py` emits **its own** `eval_results.json` +
   `eval_results_table.csv` (via `lora_remotesense/evaluate.py:370-383`) plus per-sample
   `generated/<key>.npy`, `gt/<key>.npy`, `meta/<key>.json` — **not** the record contract.
   Write an adapter that reads its per-sample output and assembles `RECORD_KEYS`, then returns
   `validate_record(record)` (`external_stubs.py:138-141`).
   - **Inode warning:** that pipeline writes **3 files per sample** (`evaluate_mvsplat360.py:431`,
     `436`, `451`). `rx40` is inode-tight. Stage them on jobfs and aggregate, or bound
     `max_samples`; do not let it scatter thousands of files onto `rx40`.
7. **Keep the black-frame tripwire.** On per-sample failure, `evaluate_mvsplat360.py` fills
   **black frames** (`:350`) and keeps computing metrics, flagging `generation_success`
   (`:449`). `p6_mvsplat.sh 第 4 步黑帧自检(现 :108-124)` enforces `generation_success >= 80%`. **Numbers obtained by
   bypassing that check are worthless and must never reach the paper.**
8. **Flip `status` → `implemented`** in the adapter, then run the self-check (exit 0).
9. **Run it** via `baseline_eval_template.pbs` with an **independent `WORK`**
   (e.g. `/scratch/rx40/$USER/RSWM/work_baselines`). Never the old AID `work` root — the
   template refuses anyway (`:45-50`).
10. **Only now** update `baseline_status.json` (`tested=true`, `result_available=true`,
    `blocking_dependency=null`) together with the two tripwires named in §8, in one commit.
11. **Write the paper row** with the footnotes: 5 context views + poses; separate env;
    `generation_success` rate; sample count.

### Gotchas — verified while writing this doc

- **`evaluate_mvsplat360.py` has live decommissioned-`mv44` argparse defaults**
  (~~`:478` `--mvsplat360_dir`, `:481` `--checkpoint`, both `/g/data/mv44/...`~~
  —— v2.4.3 已改为必填、无默认路径), and `mv44`
  paths in its usage docstring (`:11-14`). **Always pass these flags explicitly**; running it
  bare silently targets dead storage. (`p6_mvsplat.sh 推理段(现 :99-106)` does pass them explicitly.)
- **`lora_remotesense/mvsplat360_baseline.pbs` must not be reused**: its `#PBS -P` project,
  its `storage=` directive, and every data/env/output path in it point at the decommissioned
  **mv44** project (verified at `:3`, `:10`, `:35-53`). Paths are not reproduced here — none
  of them resolve. Use `server_run/pbs/baseline_eval_template.pbs` instead.
  (The `mv44` strings inside `server_run/baselines/` are only *warnings about* that file —
  `external_stubs.py:88-89`, `baseline_status.json:49` — not live paths.)
- **`~4-8 GPU-hours / 20 samples`** for MVSplat360 is the *script author's estimate*
  (`p6_mvsplat.sh` header; old PBS `walltime=04:00:00`). **Never measured in this repo.** Do
  not cite it as a measurement (`external_stubs.py:110-111`).
- **`__init__.py:5` says this file lives at the repo root**; it is actually at
  `server_run/baselines/README_BASELINES.md`. Stale pointer in a read-only file — noted, not fixed.

---

## 10. Non-negotiables

1. **No fabricated numbers.** Not for GPU hours, not for a baseline that "would probably score ~X".
2. **`gt_oracle` and `deterministic_translation` are upper bounds, never methods.** Never rank
   them against real methods.
3. **Diagnostic ≠ external comparison.** Four passing diagnostics prove the *metric
   discriminates*. They prove nothing about how RSWM compares to prior work.
4. **`tested=false` means untested.** A dry run, a `py_compile`, and a wiring self-check are
   not tests of behaviour.
5. **Old AID results are sacred.** New baselines use an independent `WORK` root; `server_run/work/`
   is never written.

---

## 11. Paper-facing v2 platform (2026-08-16 extension)

The existing adapter runner above remains the numerical implementation.  The repository-root
entry point now adds the frozen sample protocol, baseline cards, standard result layout, and table
export without changing any model:

```bash
python run_baseline.py --list

python run_baseline.py \
  --method persistence \
  --manifest manifests/aid_test.json \
  --output results/persistence
```

Formal manifests are built with
`server_run/baselines/build_benchmark_manifest.py`.  They record sample provenance,
checksums, T={17,25,33}, condition-to-future overlap curves, and distribution-derived
difficulty bins.  `sample_keys` now takes precedence over legacy split parsing; this is required
to compare every method on exactly the same test IDs.

Every successful run also emits:

```text
manifest.json
config.json
baseline_card.json
sample_metrics.jsonl
aggregate_metrics.json
videos/
logs/
```

Visual metrics exclude the condition frame (`t=0`).  Missing optional metrics are `null`, never
zero.  The torchvision video metric is exported only as `FVD_R3D_PROXY`; standard I3D output is
stored separately as `FVD_I3D`.

Lifecycle truth lives in `baseline_status.json` v2.  `implemented/configured/smoke_tested/
gpu_tested/trained/evaluated/result_available` are separate fields.  As of this extension only
Persistence has a synthetic CPU end-to-end smoke; no external/learned baseline has a paper result.
