# Command Line Interface

GraphLD provides a `graphld` CLI for LDGM-backed workflows and a separate `estest` CLI for enrichment score tests.

Use this page as the shared overview. For subcommand-specific details, follow the topic pages below.

For the enrichment score test CLI, see [Enrichment Score Test](score_test.md).

## Topics

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

- LDSC `.sumstats` format
- GWAS-VCF `.vcf` format
- Parquet `.parquet` format

GraphLD uses LDGM metadata to locate LD blocks. By default this is:

```text
data/ldgms/metadata.csv
```

Override it with `--metadata` when needed.

For annotation-driven workflows, GraphLD supports:

- LDSC `.annot` files
- UCSC `.bed` region annotations
- GMT gene-set annotations via `--gene-annot-dir`

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

`surrogates` also requires `--population`, because surrogate lookup depends on the reference population.

## Shared Output Pattern

Commands that consume summary statistics typically write a single tab-separated output file or HDF5 artifact to the `OUT` path you pass on the command line.

The main exception is `reml`, which writes multiple result files with a shared prefix.

## Choosing A Command

- Use [`reml`](cli/reml.md) for heritability partitioning and enrichment estimation.
- Use [`blup`](cli/blup.md) for infinitesimal-model effect-size weights.
- Use [`clump`](cli/clump.md) for thresholding and LD pruning.
- Use [`simulate`](cli/simulate.md) for synthetic GWAS summary statistics.
- Use [`surrogates`](cli/surrogates.md) to precompute missing-variant substitutions for repeated graphREML runs.
