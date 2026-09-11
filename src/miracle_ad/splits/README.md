# Fixed split artifacts

These files are derived from the MIRACLE-AD research workspace and intentionally
tracked for transparency. Each JSON explicitly stores `train`, `val`, and
`test`. The old `val` holdout is preserved unchanged as `test`; a deterministic,
label-stratified 10% of the old training pool (seed 42) is committed as `val`.
Legacy path prefixes and separators are normalized only at load time by
`miracle_ad.split_registry`.

`split_provenance.json` records the seed, selection procedure, and aggregate
before/after counts.

See `docs/DATASETS.md` at the repository root for schema, counts, label mapping,
and validation commands.
