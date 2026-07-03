# Enrichment Score Test

Use `estest` for score-test workflows:

```bash
uv run estest --help
uv run estest show path/to/scores.h5
uv run estest test path/to/scores.h5 path/to/output/file/prefix --variant-annot-dir path/to/annot_dir/
uv run estest test path/to/scores.h5 path/to/output/file/prefix --gene-annot-dir path/to/gmt/files/ --gene-table data/genes.tsv
```

The command also supports:

```bash
uv run estest add-meta path/to/scores.h5 meta_name trait1 trait2
uv run estest rm path/to/scores.h5 pattern -f
uv run estest mv path/to/scores.h5 old_name new_name
uv run estest convert path/to/scores.h5 path/to/gene_scores.h5
```

Use `uv run src/score_test/score_test.py --help` for the legacy standalone score-test script surface.

`--perturb-annot` is available on the legacy standalone script surface (`uv run src/score_test/score_test.py ... --perturb-annot`), not on the packaged `estest test` subcommand.
