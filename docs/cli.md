# Command Line Interface

GraphLD provides a `graphld` CLI for LDGM-backed workflows and a separate `estest` CLI for enrichment score tests, which operate on precomputed scores instead of LDGMs. For the enrichment score test CLI, see [Enrichment Score Test](score_test.md).

## Subcommands

- [graphREML (`reml`)](cli/reml.md)
- [BLUP (`blup`)](cli/blup.md)
- [LD Clumping (`clump`)](cli/clump.md)
- [Simulation (`simulate`)](cli/simulate.md)
- [Surrogate Markers (`surrogates`)](cli/surrogates.md)

## Shared Usage

Inspect the command surface with:

```bash
uv run graphld -h
uv run graphld <subcommand> -h
```

Most analysis subcommands follow this shape:

```bash
uv run graphld <subcommand> SUMSTATS OUT [options]
```

`simulate` is the exception: it takes an output summary-statistics path rather than an input summary-statistics file.

## Shared Inputs

The main `graphld` analysis commands accept summary statistics in:

- [LDSC `.sumstats` format](file_formats.md#ldsc-format-sumstats)
- [GWAS-VCF `.vcf` or `.vcf.gz` format](file_formats.md#gwas-vcf-format-vcf-or-vcfgz)
- [Parquet `.parquet` format](file_formats.md#parquet-format-parquet)

GraphLD uses LDGM metadata to locate LD blocks. By default this is:

```text
data/ldgms/metadata.csv
```

Override it with `--metadata` when needed.

For annotation-driven workflows, GraphLD supports:

- [LDSC `.annot` files](file_formats.md#ldsc-format-annot)
- [UCSC `.bed` region annotations](file_formats.md#bed-format-bed)
- [GMT gene-set annotations](file_formats.md#gmt-format-gmt) via `--gene-annot-dir`

See [File Formats](file_formats.md) for details.

## Shared Options

Many subcommands share these options:

| Option | Description |
|--------|-------------|
| `-p, --population` | Restrict LDGM selection to a population such as `EUR` or `EAS` |
| `-c, --chromosome` | Restrict analysis to one chromosome |
| `-n, --num-samples` | Override sample size when it cannot be inferred from the input |
| `--num-processes` | Set the worker-process count |
| `--run-in-serial` | Disable multiprocessing |
| `--metadata` | Use a non-default LDGM metadata CSV |
| `-v, --verbose` | Print detailed progress |
| `-q, --quiet` | Suppress normal progress output |

`surrogates` is population-specific. The CLI defaults to `EUR`; pass
`--population` explicitly for other reference populations.

## Shared Output Pattern

Commands that consume summary statistics typically write a single tab-separated output file or HDF5 artifact to the `OUT` path you pass on the command line.

The main exception is `reml`, which writes multiple result files with a shared prefix.
