# Development Log — RS-Roll Baseline Extension
> Created: 2026-08-16 | Last updated: 2026-08-16
> Related guide: `docs/implementation.md`

## Project overview

| Item | Value |
|---|---|
| Research focus | Fair benchmark for trajectory-conditioned remote-sensing video rollout |
| Strategy | Incremental extension of the current RS-Roll codebase |
| Framework | Existing Python/PyTorch stack |
| Local execution | CPU/static only; no GPU, training, download, or checkpoint load |

## Implementation progress

| Module | Main files | Status | Note |
|---|---|---:|---|
| Baseline audit | `reports/BASELINE_AUDIT.md` | Done | Registry/status/checkpoint/result truth audited |
| Protocol | `baselines/protocol.py`, manifest/leakage tools | Done | CPU verified |
| Adapters/cards | `result_adapters.py`, `cards/*.json` | Done | Bridges do not imply model results |
| Unified runner | `run_baseline.py`, `server_run/run_baseline.py` | Done | Dry-run/list/import verified |
| Evaluator | `evaluate_baseline_results.py` | Done | Future-only; optional heavy metrics |
| Experiment configs | matched 2x2 and rank sweep | Done | YAML semantics regression-tested |
| Exporter | `tools/export_paper_tables.py` | Done | Null-preserving CSV export tested |
| GPU validation | `GPU_SMOKE_TEST.md` | Pending A800 | Not run locally |
| Formal results | `results/` | Pending A800/assets | No numbers fabricated |

## Development entries

### 2026-08-16 — Audit and protocol freeze

- Audited every registry entry, lifecycle flag, checkpoint/result availability, output
  format, manifest path, and evaluator behavior.
- Found that all original methods had `tested=false` and `result_available=false` and
  that no bundled baseline checkpoint or paper result existed.
- Added explicit v2 schemas, checksums, per-frame overlap profiles, empirical bins,
  and cross-dataset leakage checks.

### 2026-08-16 — Adapter, runner, evaluator, and experiment matrix

- Added result-directory bridges for external/internal producers without changing any
  upstream model implementation.
- Added baseline cards and lifecycle distinctions.
- Fixed legacy visual scoring to exclude condition frame 0 and fixed explicit manifest
  sample keys taking precedence over split parsing.
- Added standardized result directories, future-only evaluator, I3D interface,
  matched 2x2/rank configs, and automatic paper-table export.

### 2026-08-16 — Verification and release preparation

- New platform tests: 13/13 PASS; full `tests/` discovery: 50/50 PASS.
- All runnable pre-existing CPU protocol suites passed after rerunning tempfile-based
  scripts outside the Windows filesystem sandbox.
- Full repository `compileall` passed.
- Final audit fixed float `[0,1]` normalization, condition/rollout resolution identity,
  manifest checksum ordering, manual metric injection, distribution-metric N gating,
  bridge producer identity checks, and fail-closed table export/standardization.
- LPIPS known-answer test skipped because the optional package/weight is absent.
- Shell `bash -n` not run because this host has no usable Bash runtime.

## Known issues

- Real A800 checkpoint smoke and all formal baseline/model runs remain pending.
- Standard I3D extractor, LPIPS asset, SD2 inpainting, MVSplat360, and formal RS-Roll
  checkpoints are not bundled.
- RSWISE/RemoteBAGEL public reproducibility remains blocked; do not report a result.

### 2026-08-16 — ICLR main plan and A800 wrappers

- Found that the earlier research-only matched/rank YAMLs contain metadata keys not
  accepted by the formal runner's fail-closed YAML schema. Preserved those files as
  protocol artifacts and generated a separate runner-compatible formal matrix.
- Froze training seeds 42/43/44 while holding split/generation seed 42, data,
  checkpoint, budget, evaluation count, and A800 profile fixed.
- Deduplicated the canonical native+attn_cam+r16+s42 point across primary, matched,
  and rank groups, yielding 12 rather than 14 redundant formal runs.
- Added safe plan/server-dry-run/execute wrappers. No GPU, training, downloads,
  checkpoint loads, or formal result generation occurred during this change.
- Final CPU regression: 60/60 tests passed; plan validation, legacy ICLR matrix
  validation, compileall, and the source hard-coded-GPU safety scan passed.
