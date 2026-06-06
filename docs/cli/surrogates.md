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
- [`.vcf`/`.vcf.gz`](../file_formats.md#gwas-vcf-format-vcf-or-vcfgz)
- [`.parquet`](../file_formats.md#parquet-format-parquet)
- [`.snplist`](../file_formats.md#snp-list-format)

## Population

`--population` selects the reference population for surrogate lookup and defaults to `EUR`.

## Output

The command writes an HDF5 file that can be passed back to `graphld reml` via `--surrogates`.

Use the shared runtime and metadata options from the [CLI overview](../cli.md) as needed.
