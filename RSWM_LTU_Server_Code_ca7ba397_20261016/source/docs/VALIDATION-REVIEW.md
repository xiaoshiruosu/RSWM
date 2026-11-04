# Incremental validation: 2026-09-12

Base: `7f098419cf22e93aea22cd17413af288d8e1d377` in `yedda-tv/rswm-1`.
The base GitHub Actions run `34699322002` completed successfully. Its result is not reused as evidence that this revision's new tests passed.

## What was actually run locally

`PYTHONPATH=src python -m pytest -q tests/test_review_hardening.py --junitxml=evidence/review-tests.xml`

**57 passed in 15.82 seconds.** Python 3.13; synthetic data only. This is the new targeted test module, not a claim that all prior remote tests were rerun locally. Five base modules were reconstructed and checked byte-for-byte against their Git blob SHAs before modification; see `evidence/base-files.json`.

Checks cover manifest structure and full-inventory top-k scope; self-rehashed invalid rows; city/role/path/version/rank/ID/size consistency; download caps; uint64/int64 overflow; source-path provenance; same-start exact/offset generation; F3 order after canonical JSON serialization; and independent disk replay for F2/F3/F5/F7.

Disk replay rejects changed PNG pixels, deleted valid outputs, unlogged outputs, failed-slot replacements, broken chains, rehashed wrong coordinates, missing requests/responses/outputs, duplicate outputs, changed task specifications, wrong external heads, and changed summaries. Duplicate JSON keys and non-finite JSON constants are rejected.

The positive replay fixture contains 352 new outputs across F2/F3/F5/F7. Another F7 fixture has 160 planned outputs, of which the first 16 are missing: 144 are valid, but target first references remain invalid and paired reentry MAE remains null. A synthetic 5000-square TIFF exercises conversion and source-path preservation; it is not an Inria download.

## Supplementary paired-geometry audit

24 synthetic world identifiers, 96 origin groups, 384 supplementary packages and 49,152 planned new outputs were generated and geometrically validated. Minimum observed invisibility gap was 62; the specification still requires only 32. Joint proposal counts ranged from 1 to 2. The direction histogram and runtime are recorded in `evidence/paired-geometry-review.json`.

This audit generated queries and labels, not 49,152 model images. It does not add packages to the 1,920-package core. Real source-manifest IDs will produce their own deterministic samples and must be audited again.

## Trust and scientific limits

An unkeyed digest or internally valid journal is not authentication. Full-log replacement is detectable only with an independently retained trusted head. File replay does not verify network isolation, hidden-input exclusion, actual model snapshot isolation, trustworthy timing, GPU use, or a malicious adapter's causal behavior. This implementation is a cooperative reference runner, not a public-evaluation sandbox.

No real TIFF, complete HF revision/inventory, pretrained model weights, GPU inference, learned numerical conditioning, real-image motion calibration, or frozen feature-quality experiment was obtained in this incremental validation. Existing generic motion/MMD functions are not certified quality measurements.

## Reproduction

```bash
python -m pip install -e '.[dev,data,motion]'
python -m pytest -q
python -m rswm_inria.cli demo --model oracle --save-images --out runs/oracle-new
python -m rswm_inria.replay --run-dir runs/oracle-new/synthetic-demo-o0-F5 --out runs/oracle-new/f5-recomputed.json
```

New runs save evaluator-private `task.json` and `initial.png` and bind the task digest in `rswm-journal-v1`. Old journals lacking this information are not silently certified as the new format. The source task version and random namespace remain unchanged.

The final remote commit and CI result must be read from GitHub; a later delivery receipt records that result without changing this local-evidence claim.
