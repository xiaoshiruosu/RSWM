# Independent paired and native LR64 execution

These entry points execute, replay, and aggregate their own frozen plans. Neither study is added to the Inria core leaderboard. Each variant resets the adapter independently; first-attempt RGB results remain primary, and recovery results and every started attempt's costs are reported separately.

## Same-start paired study

`rswm_inria.paired` generates exact/offset tasks from a frozen Inria manifest. Its index must be associated with the materialized world index; a task's public initial crop, world bytes, role, origin, protocol, target, and first 80 queries are checked before execution.

```bash
python -m rswm_inria.paired --manifest build/manifest.json --out build/paired-tasks --role test_unseen_city
python -m rswm_inria.paired_execution plan \
  --task-index build/paired-tasks/index.json \
  --world-index build/materialized/private/world-index.json \
  --models configs/local-models.json --seeds 0 \
  --code-commit COMMIT_SHA --out build/paired-plan.json
python -m rswm_inria.paired_execution run --plan build/paired-plan.json --root build/paired-suite
python -m rswm_inria.paired_execution verify --plan build/paired-plan.json --root build/paired-suite
python -m rswm_inria.paired_execution aggregate \
  --plan build/paired-plan.json --root build/paired-suite --out build/paired-analysis.json
```

Select a bounded pilot index before reading generated outputs. The complete paired extension adds 49,152 new outputs per method and seed, and these commands do not silently approve that budget.

`--mechanisms` accepts a JSON object mapping model IDs to frozen policies. A policy contains exactly `schema: rswm-history-policy-v1`, `mechanism`, `capacity`, `protect_initial`, and `protect_history`. The mechanism is `last_frame`, `frame_cache`, or `map_assisted`; LastFrame capacity is exactly one. The corresponding model config uses `adapter: lazy-map`, with local completer weights or explicitly diagnostic noise. Its model ID ends in `-` followed by the first 16 hexadecimal characters of the policy digest. Thus policy changes produce a different frozen model identity.

LastFrame retrieves the last successfully generated frame, FrameCache selects the retained frame with greatest viewport overlap and breaks ties by recency, and MapAssisted reads its generated spatial map. I0 and generated-history protection are separate switches. Retrieval positions, hits, pixels, completion/native-frame calls, commits, and cache evictions are measured. Snapshots preserve logical history and deterministic request seeds; cumulative costs never roll back across F7 branches. These wrappers are mechanism experiments and do not establish reproduction of a trained backbone.

The auxiliary `first-reference-final-target-v1` score uses the same immutable first reference and final PNGs as replay. Primary paired statistics retain missing sides and report common-pixel coverage. A world becomes complete only when all planned pair/origin/seed cells in its role/family are complete. Bootstrap resamples complete worlds with every dependent pair and seed retained. Offset changes geometry, so exact-minus-offset is not a pure causal memory estimate.

## Native LEVIR-LR64 study

The pinned materializer reads only image1 from the three locked test Parquet shards and preselects 100 of 348 rows by the existing study/shard/row hash. It requires the actual shard bytes and hashes. Execution validates the materialized source lock, selected-row provenance, public I0/session, original world shape/pixels, and every private geometry label.

```bash
python -m rswm_inria.lr64 plan --materialized build/levir100-lr64 \
  --models configs/lr64-models.json --seeds 0 --code-commit COMMIT_SHA \
  --out build/lr64-plan.json
python -m rswm_inria.lr64 run --plan build/lr64-plan.json --root build/lr64-suite
python -m rswm_inria.lr64 verify --plan build/lr64-plan.json --root build/lr64-suite
python -m rswm_inria.lr64 aggregate \
  --plan build/lr64-plan.json --root build/lr64-suite --out build/lr64-analysis.json
```

Native worlds remain 1024×1024 RGB uint8 throughout. The public viewport is 240×416, query bounds are ±288, and each variant receives four online batches of 16 queries. Exact/offset share their first 48 queries and independently reset. The center-128 target at B is scored at its geometrically computed first reentry and at event 64. A missing first planned exposure stays invalid permanently, including at the final target. The LR64 runner, 1024 ledger, journal, replay, plan schema, and output identities are independent of core F5/F7.

Native diagnostic adapters are `copy`, `random`, `native-noise-map`, and evaluator-only `oracle`. They execute on CPU and cannot record unused CUDA configuration. `history-completer` accepts a frozen history `policy` and local completer weights, or explicitly diagnostic noise. Any other adapter must explicitly support the native world specification; the runner refuses unsupported dimensions rather than resizing or padding.

For a small synthetic pilot, `plan --task-index` instead accepts `rswm-lr64-diagnostic-index-v1`, the independent study ID, `diagnostic_only: true`, a frozen source hash, and relative task/world paths. Every indexed task must have role `diagnostic_only`. This path never claims the real LEVIR100 data gate.

## Retry, faults, and evidence

Both plan CLIs default to one first attempt. A larger finite budget must be frozen with `--max-attempts-per-run`; later attempts require `run --resume --retry-failed`. Default permitted retry statuses are load failure, interrupted execution, or process crash. Native hard process crashes are closed as `CRASHED` on resume, with unmeasured old costs retained as unknown. Recovery never restores hidden state from generated PNGs or changes the primary first-attempt sample.

Exit code 0 means the first-attempt submission and all started-attempt resource costs satisfy integrity checks. Code 1 writes useful analyses with failed/missing runs or incomplete/corrupt per-run metadata. Code 2 rejects invalid root plans or arguments. A corrupt resource JSON preserves independently verified RGB; a corrupt outcome retains the full frozen expected denominator; a corrupt root does not produce an accepted analysis.

```bash
python scripts/g7_diagnostics.py --out evidence/build-closure/G7-cli-final
```

This executes actual CLIs on one synthetic world per study, including resource/outcome/root-plan fault cases, and saves commands, return codes, hashes, and result accounting. The directory must be new. It aborts if source code changes during execution. Its Oracle checks validate engineering only: genuine source data, local pretrained weights, GPU performance, and geographic independence remain separate NOT_RUN/BLOCKED gates until real evidence is supplied.
