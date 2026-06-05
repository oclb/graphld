# BLUP (`blup`)

Use `graphld blup` to compute infinitesimal-model effect-size weights.

## Basic Usage

```bash
uv run graphld blup \
    /path/to/sumstats.sumstats \
    output.tsv \
    --heritability 0.1 \
    --num-samples 100000
```

## Required Option

| Option | Description |
|--------|-------------|
| `-H, --heritability` | Heritability for the analyzed variant scope, between `0` and `1` |

## Inputs And Output

- Input summary statistics: `.sumstats`, `.vcf`, or `.parquet`
- Output: a tab-separated file of BLUP weights at the path you pass as `OUT`

BLUP also uses the shared LDGM options documented in the [CLI overview](../cli.md).

## Notes

- BLUP assumes an infinitesimal effect covariance matrix `D`; GraphLD sets `trace(D)` equal to `--heritability` across the matched LDGM effect indices in the analyzed scope. If you restrict BLUP with `--chromosome` or `--population`, pass the heritability for that restricted scope.
- If `--num-samples` is omitted, GraphLD infers sample size from the input file.
- Use `--population`, `--chromosome`, and `--run-in-serial` to limit the scope of a run or simplify debugging.
