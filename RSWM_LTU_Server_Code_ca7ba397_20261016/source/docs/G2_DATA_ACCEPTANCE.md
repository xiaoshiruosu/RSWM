# G2 data engineering and acceptance

The source selector, city roles, central crop, 2×2 half-up rounding, task queries,
first planned exposure and RGB scoring are preserved. Construction now rejects
files whose actual container is not a single-page TIFF with three unsigned
8-bit RGB samples. Every decoded source requires its frozen SHA256 and size.
The default revision is `ad8315c82c8a3021b93a7d52922d19ea76201f60`; mutable
revisions are rejected before network access, and an offline inventory must
match the explicitly requested revision.

These commands need the actual source lock and local source TIFFs. They do not
implicitly download all sources or any model weights:

```bash
python -m rswm_inria.cli freeze --revision ad8315c82c8a3021b93a7d52922d19ea76201f60 --out data/inria-lock/manifest.json
python -m rswm_inria.cli first-source --manifest data/inria-lock/manifest.json --inventory data/inria-lock/source-inventory.json --source-dir data/inria-raw --out build/inria-first-source
python -m rswm_inria.hf_materialize inria --manifest data/inria-lock/manifest.json --inventory data/inria-lock/source-inventory.json --profile smoke12 --source-dir data/inria-raw --out build/inria-smoke12 --max-source-gb 1.0
python -m rswm_inria.materialized --build build/inria-smoke12 --manifest data/inria-lock/manifest.json --inventory data/inria-lock/source-inventory.json --source-dir data/inria-raw --out evidence/inria-smoke12-audit.json
python -m rswm_inria.smoke12 --build build/inria-smoke12 --manifest data/inria-lock/manifest.json --inventory data/inria-lock/source-inventory.json --source-dir data/inria-raw --out runs/inria-smoke12-diagnostics --code-commit FULL_CURRENT_GIT_SHA --max-output-gb 16 --execute
```

Omit `--execute` on the last command to freeze the complete diagnostic plan and
inspect its explicit disk budget; that result is `PLANNED_NOT_RUN`. The full
execution requires 192 tasks per method and verifies 16,896 unique generated
outputs per method, 576 runs and 50,688 outputs across Oracle, Copy and
StaticNoiseMap. StaticNoiseMap is `noise-map`, distinct from `lazy-noise`.
Oracle's applicable RGB errors must be zero with full coverage. Copy's error
is measured rather than hard-coded. StaticNoiseMap's consistency errors must
be zero. Every output is replayed through the existing suite; resource coverage,
world/city tables, first/recovery selections, failure distribution and derived
hashes remain in the acceptance directory.

The auditor recomputes source→world pixels using four independent strided
slices, checks all four real I0 crops per world, verifies the complete file
inventory and public session whitelist, compares frozen task coordinates, and
reconstructs every private target-pixel label from event-by-event visibility.
Private NPZ keys, integer shapes and hashes are all checked. Re-encoding an NPZ
with identical arrays preserves semantic identity after its declared container
hash is updated. Duplicate world pixels require review; they are never silently
replaced. Geographic independence remains unknown when geospatial evidence is
absent. SHA256 identity does not establish source authenticity or a license.

New materialization exports labels under `private/tasks/labels` with task-local
`labels/...` references. This satisfies the suite's confinement rule without
allowing `..` paths. Old exports using `private/labels` and `../labels` need an
explicit new-directory rebuild; they are not silently rewritten or accepted by
weakening the suite guard. All output directories must be empty.

`materialized.validate_audit_receipt(report, plan, required_profile=...)` binds a
current real-data audit to the planned source manifest, code fingerprint, world
file/pixel hashes, city/role and task semantic hashes. Synthetic fixtures and
thin `PASS` dictionaries cannot authorize a real pilot or formal experiment.
The receipt is provenance, not a cryptographic signature.

In the original environment, the resumed build had the frozen real inventory/manifest
and all twelve smoke12 TIFFs (900,605,981 source bytes). Its complete materialization passed the
independent pixel, query, label and artifact-inventory audit; its receipt is
`evidence/build-closure/G2-real-resumption-encoded/materialized-audit.json`.
Complete three-method execution is tracked separately in
`evidence/build-closure/G2-real-full-smoke12/`; a materialized audit alone does
not establish diagnostic execution acceptance. The three earlier failed build
directories and post-unlink file reintroduction evidence are retained. The
unknown writer has not been identified. Full real smoke12 finished with exit 0,
`REAL_DATA_VALIDATED` / `ACCEPTED`: 576 runs and 50,688 unique outputs, with
192 runs / 16,896 outputs / 192 complete resource receipts per diagnostic model.
Lossless acceptance, independent replay, metadata and table evidence can be
restored from `evidence/build-closure/G2-real-full-smoke12/collected-for-git/`.
The original plan retains its actual fd7-based dirty source identity (package
fingerprint 9ef); it is not relabeled as a clean later commit. Later motion
search-domain metadata corrections change the package source identity even
though the existing integer overlap predicate, task geometry and RGB scorer
remain unchanged. The original real smoke12 acceptance and G5 validation stay
bound to their original 9ef source; they are not whole-package validation of the
corrected source. Fresh runs must retain source-matching plans and audit
receipts rather than bypassing source fingerprint guards. Exact software
checkpoint and installed-archive evidence is recorded in the
[resumption delivery report](../evidence/build-closure/RESUMPTION_DELIVERY_2026-09-30.md).

The original 13:57 core120 preflight recorded 36 SHA-verified cached sources
and 84 missing, requiring 9,014,030,277 bytes with 6,391,939,072 bytes available.
Those cache counts and free-space readings are historical after environment
recovery; a fresh complete cache/space preflight is required before execution.
At that historical preflight, download, materialization and independent core
audit were `NOT_RUN` / `BLOCKED_SPACE`. Recovered-environment execution and
independent-audit outcomes require their new actual receipts, tracked in the
resumption delivery report and the exact handoff in
`evidence/build-closure/G6-core120-real-resumption/HANDOFF.md`. Two post-run
hardlink storage operations aborted on inode/link stability guards; all 57,634
original file paths and SHA256 bytes were independently verified unchanged
after each failure. Failed ledgers, staging links and reserves are retained,
further hardlink attempts are stopped, and block dedupe was unsupported in the
probed filesystem scope. No reclamation estimate authorizes core120. Real
generation models, GPU, original source rights and geographic independence
remain unaccepted.

The earlier synthetic test's 12 runs and 1,056 outputs remain an engineering
subset. They are never promoted to full real smoke12 or paper experiments.
