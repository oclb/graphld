# Enrichment Score Test

Use `estest` for score-test workflows:

```bash
uv run estest --help
uv run estest show STATISTICS.h5
uv run estest test STATISTICS.h5 OUT_PREFIX --variant-annot-dir ANNOT_DIR
uv run estest test STATISTICS.h5 OUT_PREFIX --gene-annot-dir GMT_DIR --gene-table data/genes.tsv
```

The command also supports:

```bash
uv run estest add-meta STATISTICS.h5 META_NAME TRAIT1 TRAIT2
uv run estest rm STATISTICS.h5 PATTERN -f
uv run estest mv STATISTICS.h5 OLD_NAME NEW_NAME
uv run estest convert VARIANT_STATS.h5 GENE_STATS.h5
```

Use `uv run src/score_test/score_test.py --help` for the legacy standalone score-test script surface.

`--perturb-annot` is available on the legacy standalone script surface (`uv run src/score_test/score_test.py ... --perturb-annot`), not on the packaged `estest test` subcommand.
