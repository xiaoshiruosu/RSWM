# Baseline Extension Test Report

Date: 2026-08-16  
GPU accessed: **NO**  
Training started: **NO**

## Final local verification

`python tests/test_baseline_platform.py -v`: **PASS (13/13)**

Covered:

- registry/card parity;
- v2 manifest and per-frame overlap profile;
- data/split/sample checksums;
- empirical difficulty bins;
- 17/25/33-frame clips;
- heterogeneous condition-image sizes;
- Persistence generate -> standard output -> evaluator;
- future-only scoring and condition-frame exclusion;
- missing optional metrics remain null;
- unavailable external baseline fails loudly;
- empty result bridge fails loudly;
- malformed/empty manifest rejection;
- cross-dataset exact-duplicate detection;
- matched 2x2 config invariants;
- native-attn_cam rank sweep invariants;
- automatic paper-table export;
- frozen-manifest tamper detection and diagnostic subset/hash-skip downgrade;
- producer condition-identity rejection and exporter row-count fail-close;
- distribution-metric minimum-N gate before feature-model loading.

`python server_run/baselines/registry.py`: **PASS**

- 16 registered adapters;
- status JSON/registry reconciliation;
- record key/shape/dtype validation;
- lifecycle distinction retained.

`python server_run/iclr_matrix.py --check-only`: **PASS**

- 12/13 planned runs marked runnable;
- full fine-tuning remains explicitly blocked/not implemented.

`python -m unittest discover tests -v`: **PASS (50/50)**

- includes the 13 baseline-platform tests above;
- experiment-runner and A800 free-memory mocks;
- shared-GPU scheduler mocks;
- RESISC45 CPU decoding/preprocessing;
- WHU-RS19 CPU decoding, preprocessing, snapshot resolution, and integrity fixtures.

Repository CPU protocol scripts: **PASS**

| Command | Result | Scope |
|---|---:|---|
| `python server_run/test_a800_profile.py` | PASS | A800 configuration/static policy |
| `python server_run/test_backbone_protocol.py` | PASS (86/86) | TrivialShift real CPU path; Wan path mocked |
| `python server_run/test_iclr_protocol.py` | PASS (25/25) | matrix, manifest, split, future-only rollout |
| `python server_run/test_metrics_known_answer.py` | PASS (68/68), SKIP (1) | known-answer metrics; LPIPS dependency absent |
| `python server_run/test_pose_timeline.py` | PASS (297/297) | pose/trajectory/timeline protocol |
| `python server_run/test_route_resolution.py` | PASS (15/15) | route metadata resolution |
| `python server_run/test_submission_contract.py` | PASS (93/93) | strict output and future-only contract |
| `python server_run/test_semantic_gate.py` | PASS (88/88) | injected scorer and CPU H5 fixtures |
| `python server_run/test_taxonomy_prompt.py` | PASS (49/49) | taxonomy/prompt invariants |
| `python server_run/test_wellposed_dualcolumn.py` | PASS (17/17) | trajectory well-posed reporting |
| `python server_run/baselines/external_stubs.py` | PASS (6/6) | output-bridge validation only, not upstream MVSplat360 |

The Windows managed sandbox denies access to directories created by Python's
`tempfile` ACL path.  Five pre-existing scripts were therefore rerun outside the
filesystem sandbox, still using CPU only.  They passed without source changes.

`python -m compileall -q .`: **PASS**

CLI import/help checks for the unified runner, evaluator, manifest builder, leakage
checker, and table exporter: **PASS**.

`bash -n` over shell scripts: **NOT RUN**.  This Windows host exposes only the WSL
application alias; `bash --version` fails with `Bash/Service/CreateInstance/E_ACCESSDENIED`,
and Git Bash/MSYS2/Cygwin are not installed.  No shell script was executed.

## Deliberately not run

- real image-outpainting checkpoint;
- real frozen/no-camera/RS-Roll inference;
- standard I3D checkpoint;
- any GPU smoke or formal experiment.

No training, GPU allocation, download, checkpoint load, or formal benchmark evaluation
occurred during this extension.  Consequently all paper result metrics remain unavailable.
