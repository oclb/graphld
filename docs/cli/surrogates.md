# Surrogate Markers (`surrogates`)

Use `graphld surrogates` to precompute HDF5 surrogate-marker files for GWAS variants that are missing from LDGM references.

## Basic Usage

```bash
uv run graphld surrogates \
    /path/to/sumstats.sumstats \
    surrogates.h5 \
    --population EUR
```

## Inputs

Accepted inputs:

- [`.sumstats`](../file_formats.md#ldsc-format-sumstats)
- [`.vcf`](../file_formats.md#gwas-vcf-format-vcf)
- [`.parquet`](../file_formats.md#parquet-format-parquet)
- [`.snplist`](../file_formats.md#snp-list-format)

## Important Requirement

`--population` is required for this subcommand.

## Output

The command writes an HDF5 file that can be passed back to `graphld reml` via `--surrogates`.

Use the shared runtime and metadata options from the [CLI overview](../cli.md) as needed.
