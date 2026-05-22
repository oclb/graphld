# BLUP (`blup`)

Use `graphld blup` to compute infinitesimal-model effect-size weights.

## Basic Usage

```bash
uv run graphld blup \
    /path/to/sumstats.sumstats \
    output.tsv \
    --heritability 0.1
```

## Required Option

| Option | Description |
|--------|-------------|
| `-H, --heritability` | Heritability parameter between `0` and `1` |

## Inputs And Output

- Input summary statistics: `.sumstats`, `.vcf`, or `.parquet`
- Output: a tab-separated file of BLUP weights at the path you pass as `OUT`

BLUP also uses the shared LDGM options documented in the [CLI overview](../cli.md).

## Notes

- If `--num-samples` is omitted, GraphLD infers sample size from the input file.
- Use `--population`, `--chromosome`, and `--run-in-serial` to limit the scope of a run or simplify debugging.
