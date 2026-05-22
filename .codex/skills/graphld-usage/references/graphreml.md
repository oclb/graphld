# graphREML, LDGMs, And Precision Operations

Use graphREML through:

```bash
uv run graphld reml SUMSTATS OUT_PREFIX \
  --metadata data/ldgms/metadata.csv \
  --annot-dir ANNOT_DIR \
  -p EUR
```

Use `--run-in-serial` for small smoke tests, debugging, or O2-specific runs. Use `--score-test-filename` to write HDF5 derivative statistics for later `estest` workflows.

LDGM metadata is a CSV with rows pointing to edgelist and snplist files. Downloaded LDGMs conventionally live under `data/ldgms/`; small intended fixtures live under `data/test/`.

For API-level LDGM work, inspect `graphld.io.load_ldgm`, `graphld.io.read_ldgm_metadata`, `graphld.io.merge_snplists`, `graphld.io.partition_variants`, and `graphld.precision.PrecisionOperator`.
