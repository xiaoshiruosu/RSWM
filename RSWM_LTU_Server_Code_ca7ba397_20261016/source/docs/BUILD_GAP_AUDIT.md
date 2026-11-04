# Build gap audit

Baseline: authenticated Phase A `3f733e6880c296c43a40fe9535ab4f69e0025687`.
All 57 remote tracked blobs matched the existing local snapshot. The original
seven test modules pass 181 tests, with two optional torch skips. The repository
was restored as an exact shallow Git boundary because private Git transport has
no credentials; the authenticated connector confirms main ancestry and HEAD.

| Contract | Existing execution path | Gap and planned verification |
|---|---|---|
| D03 | build_plan → _attempt → run_task → replay_suite | v2 latest policy has no frozen budget; per-run JSON outside try; recursive source hashes, first/recovery views, strict identity and lock tests |
| D04 R01 | ResourceCollector.phase → finish → resource_totals | partial GPU scalar appears complete; v3 field units/applicability/complete propagated to tables; both sync failure orders |
| D04 R02 | resource_state → state_storage | shared memoryview owner counted twice; resolve supported roots, keep unsupported partial storage distinct; 16-byte regression |
| D04 R03 | resource_counters before/after phase | failed pre-read defaults zero, 12 becomes22; typed reads, stable epochs and unique intervals; rollback/finish tests |
| D07 | expected left join → cells → worlds → cities | preserve current estimand, unique output counts, roles/identities, seed coverage and all-attempt hashes; null and full-missing tests |
| Paper export | CSV/aggregation/provenance | no LaTeX; four automatic tables, escape untrusted model labels, partial-resource markers and integrity summary |
| D01/D02 | data.py / hf_materialize.py | no local real assets; PNG disguised as TIFF and multipage input accepted; independent materialized audit and first-item/smoke acceptance required |
| D05 | motion.estimate_translation / diagnose_sequence | image-only provisional estimator present; no frozen calibration/validation roles or suite protocol; real validation blocked |
| D06 | quality.plan_tiles / assemble_tiles / polynomial_mmd_unbiased | first ledger helper present; no frozen feature weights/reference protocol/cache pipeline; real weights absent |
| D08 | WindowMap / DiffusersCompleter / WanI2VAdapter | cooperative adapters exist; registry must execute all allowed configuration; numerical backend and training require selected original source/weights |
| D09/D10 | paired.py / LEVIR source helpers | paired tasks present; LR64 must remain independent native1024 runner/ledger/replay, never rebrand as F5 |
| D11/D12 | in-process harness / setuptools | no OS worker security validation, no clean wheel acceptance or full profile gates; cannot promote subprocess to sandbox |

Frozen research files are listed in `evidence/build-closure/frozen-files.json`.
Old audit is preserved independently under `evidence/build-closure/old-audit`;
its fixed-import reproduce script proves old failures only. New negative tests
import this working tree. Current evidence describes synthetic CPU fixtures and
mocked CUDA; there are no real imagery, generator or GPU results in this audit.

The research branch at `656ed3f` is separately fetched for later G3 review.
Its camera math/Frontier/source probes and tests cannot be accepted by adding
historical test counts. Combination regressions will run after G1 closure.
