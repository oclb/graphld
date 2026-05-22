# Other GraphLD CLI Workflows

The top-level command is:

```bash
uv run graphld -h
```

Subcommands:

```bash
uv run graphld blup SUMSTATS OUT --metadata data/ldgms/metadata.csv -H 0.1
uv run graphld clump SUMSTATS OUT --metadata data/ldgms/metadata.csv --min-chisq 30 --max-rsq 0.1
uv run graphld simulate OUT.sumstats --metadata data/ldgms/metadata.csv -n 10000 -H 0.2
uv run graphld surrogates SUMSTATS OUT.h5 --metadata data/ldgms/metadata.csv
```

Use `-p/--population`, `-c/--chromosome`, and `--run-in-serial` for small or constrained runs.
