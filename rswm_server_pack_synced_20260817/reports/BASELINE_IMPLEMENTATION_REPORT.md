# Baseline Implementation Report

Code baseline: `rsroll-bench-iclr-20260815-a800-whurs19-experiment-runner-safety-patched`  
Extension: `rsroll-bench-iclr-20260816-baseline-extension`

## Core-model change declaration

Wan/LingBot Video DiT modified: **NO**  
Video VAE modified: **NO**  
Attention / 3D RoPE modified: **NO**  
Camera injection / Plücker logic modified: **NO**

The extension is confined to baseline adapters, manifest/evaluation protocol, configs,
runner/exporter, tests, and documentation.

## Implemented platform components

- Existing registry retained and extended with result-directory adapters.
- Immutable `baseline_card.json` templates for every registered method.
- Frozen v2 benchmark manifest with per-sample checksums and overlap profiles.
- Empirical difficulty-bin derivation with unresolved-state protection.
- Cross-dataset exact/near-duplicate audit.
- Paper-facing unified runner and standardized result directories.
- Future-only visual scoring in both legacy and standardized evaluator paths.
- Explicit `FVD_R3D_PROXY` versus `FVD_I3D` naming.
- Automatic CSV export for main, overlap, trajectory, rank, target, cross-dataset,
  and efficiency tables.
- Matched pose x target configs and native-attn_cam rank sweep.

## Method lifecycle status

| Method | Implemented | Configured | Smoke tested | GPU tested | Trained | Officially evaluated | Result available |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | YES | YES | YES (synthetic CPU) | N/A | N/A | NO | NO |
| Image outpainting | adapter YES | YES | NO real checkpoint | NO | upstream only | NO | NO |
| RSWISE native method | NO | NO | NO | NO | NO | NO | NO |
| RSWISE recursive bridge | YES | YES | failure path only | NO | external | NO | NO |
| Frozen I2V | producer + bridge YES | YES | NO | NO | N/A | NO | NO |
| Same backbone no-camera | bridge/config YES | YES | NO | NO | NO | NO | NO |
| RS-Roll matched arms | bridge/config YES | YES | NO | NO | NO | NO | NO |
| MVSplat360 | output bridge only | partial | fake bridge only, not upstream | NO | external | NO | NO |

`baseline_status.json` is the machine-readable source of truth.  The meaning of
`implemented` is local code-path availability; it never implies a paper result.

## Unified commands

```bash
python run_baseline.py --list

python run_baseline.py \
  --method persistence \
  --manifest manifests/aid_test.json \
  --output results/persistence

python tools/export_paper_tables.py \
  --results results \
  --output paper_results
```

No formal experiment was started while building this extension.

