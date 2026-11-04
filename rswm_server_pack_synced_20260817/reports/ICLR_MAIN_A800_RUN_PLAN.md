# ICLR main experiment run plan — A800 80GB

Status: **CONFIGURED / CPU VALIDATED / NOT RUN**  
Date: 2026-08-16

## Scientific identity

The headline model is `native + attn_cam + LoRA rank 16`. Training seeds are
42/43/44. The source-disjoint AID split seed and generation seed remain fixed at 42,
so seed variation measures optimization/training variance rather than changing test
examples or diffusion noise. The evaluation scope is predicted frames 1 through
`T-1`; the observed condition frame is excluded.

## Canonical training matrix

| Family | Pose | Targets | Rank | Train seeds | Unique runs |
|---|---|---|---:|---|---:|
| Primary | native | attn_cam | 16 | 42,43,44 | 3 |
| Matched 2x2 additions | native/drop | attn_only/attn_cam | 16 | 42 | 3 |
| Optional all-target | native/drop | all | 16 | 42 | 2 |
| Rank additions | native | attn_cam | 4,8,32,64 | 42 | 4 |
| Total | — | — | — | — | 12 |

The matched and rank groups both reference the canonical seed-42 primary run for
their identical `native + attn_cam + r16` cell. It must not be trained twice.

All configurations use A800 batch size 1, gradient accumulation 4, BF16 through the
formal runner, five epochs, 33 maximum frames, offload enabled, 70 GiB free-VRAM
gate, and the same model/data/split/evaluation budget.

## Benchmark tracks

- Train: AID only.
- In-domain test: frozen AID test manifest.
- Test-only generalization: frozen WHU-RS19 and RESISC45 manifests.
- Main baseline set: persistence, frozen I2V, image outpainting, adapted RSWISE
  bridge, matched same-backbone no-camera, and RS-Roll.
- Missing external checkpoint/producer outputs remain blocking dependencies; the
  runner fails closed and does not synthesize substitute results.

## Execution gates

1. Plan validation must pass with the same strict parser as the formal runner.
2. Server dry-run must resolve the real model, data, split, and output paths.
3. The existing scheduler must report `READY_FOR_TRAINING` with every readiness bit.
4. The existing A800 preflight, GPU ownership lock, and complete GPU smoke chain
   remain mandatory.
5. Formal execution requires `RSWM_CONFIRM_FORMAL_RUN=YES`.

## Current evidence

- Plan/config semantic validation: **PASS (CPU)**.
- Python command materialization: **PASS (CPU)**.
- Repository CPU regression: **PASS (60/60 tests)**.
- Hard-coded GPU/source safety scan: **PASS (0 findings)**.
- Shell syntax on Linux: **NOT RUN on this Windows host**.
- Real dataset/checkpoint path check: **NOT RUN (A800 server required)**.
- GPU smoke: **NOT RUN**.
- Formal training, baseline generation, evaluation, and paper tables: **NOT RUN**.

No result number, checkpoint, or claim of model superiority was produced while
creating this plan.
