# Journal I/O investigation and publication boundary

The original local wheel acceptance at `80ba4d2` remains a failed attempt. Its
10,257-byte log (SHA256 `505debc137ce0bc7f7db60ea0316babfac8064f61211cdb68658d50e34f48247`)
is a valid 29-row prefix ending at output event25. This particular offline32 F2
needs34 scheduled request/response/output rows plus session and initialize,
thus36 rows; outputs26–32 are absent from the log. Preserved PNGs cannot repair
those rows or replace the independently retained head. The old replay receipt
and later CI/retry successes do not explain the historical inconsistency.

A bounded investigation used the same retained CI wheel, original task/world
and seed in six unique directories (three workspace, three `/tmp`). All six
passed complete32-output pixel replay and disk/memory head comparison. Both
locations resolved to the same overlay mount; their path names do not establish
independent storage. The historical cause is **not proven**, and these six runs
are not a claim of general filesystem reliability. Raw failure inputs, receipts,
logs and PNGs were verified in the retained engineering ZIP and read back from
the approved persistent files alongside the actual distribution ZIP and CI
synthetic fixture. No original attempt was edited.

Separate fault injection demonstrated defects in the old I/O state machine:
append advanced memory before failing file I/O, ignored short writes, and two
standalone Journals could claim the same empty directory before their first
append. A full F2 short-write control returned a summary while strict replay
correctly rejected its malformed journal. These are independently reproducible
engineering defects; they are not established causes of the historical failure.

The minimal repair keeps the event schema, research geometry, planned slots,
model exception handling and suite attempt selection unchanged:

- Exclusively create `events.jsonl` before the first event, including in an empty
  pre-created worker directory. Existing submissions cannot be overwritten.
- Bind its inode and committed length; a removed, truncated or replaced log
  cannot be silently recreated or continued.
- Check the binary write length, flush, synchronize and close successfully before
  advancing memory rows/head. Any uncertain I/O stops that writer permanently
  and preserves the partial prefix as failure evidence.
- Re-read and compare the actual complete log before returning a run summary.
  Core and LR64 use this same check. Independent suite replay still occurs before
  a completed outcome and retained receipt.
- Reuse the existing exclusive artifact publication primitive for PNG/task/summary
  bytes, checking short writes. Encoding failure does not expose a partial final
  PNG; a directory sync failure after publication is raised and the already
  complete artifact remains as failure evidence.

Evaluator I/O stays outside the model-exception conversion path. A failed or
interrupted attempt retains its planned denominator and observed resource costs;
a complete PNG count alone cannot sign it off. Process exit before summary/head
publication remains a crashed first attempt, without automatic recovery.

Synchronization calls and re-reading add actual evaluator I/O cost. They do not
prove power-loss durability (the observed mount advertised `fsync=volatile`),
protect against later external modification, or create an isolation boundary.
Independent externally anchored replay remains required. Before large expensive
runs in the same unproven environment, the owner must choose a validated storage
runtime or explicitly accept the bounded observation and unresolved risk.

Motion review separately recovered the exact historical candidate implementation:
its48,419 candidates equal the current set on the398,049-point integer grid.
The historical `floor(240*(1-.55))` metadata omitted `(0,±108)`; actual search
already included them. Synthetic estimator controls do not constitute new
Bellingham/Bloomington holdout exposure or formal real-model measurement.
