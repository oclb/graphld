# LD Clumping (`clump`)

Use `graphld clump` for p-value thresholding and LD-based pruning.

## Basic Usage

```bash
uv run graphld clump \
    /path/to/sumstats.sumstats \
    output.tsv \
    --min-chisq 30 \
    --max-rsq 0.1
```

## Clumping Options

| Option | Default | Description |
|--------|---------|-------------|
| `--min-chisq` | `30` | Minimum chi-squared value for index-variant inclusion |
| `--max-rsq` | `0.1` | Maximum LD threshold for pruning |

## Inputs And Output

- Input summary statistics: `.sumstats`, `.vcf`, or `.parquet`; see [Summary Statistics](../file_formats.md#summary-statistics)
- Output: a tab-separated file of retained index variants

All shared LDGM and runtime options from the [CLI overview](../cli.md) also apply here.
