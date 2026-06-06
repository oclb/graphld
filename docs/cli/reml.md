# graphREML (`reml`)

Use `graphld reml` for heritability partitioning and annotation enrichment estimation.

## Basic Usage

```bash
uv run graphld reml \
    /path/to/sumstats.sumstats \
    output_prefix \
    --annot-dir /path/to/annotations/
```

You must provide one annotation source:

- `--annot-dir` for variant or region annotations
- `--gene-annot-dir` for GMT gene sets

## Input Types

- Summary statistics: LDSC-style `.sumstats`, GWAS-VCF `.vcf`/`.vcf.gz`, or kodama-style `.parquet`; see [Summary Statistics](../file_formats.md#summary-statistics).
- Variant annotations: per-chromosome `.annot` files, optionally alongside `.bed` files; see [Annotations](../file_formats.md#annotations).
- Gene annotations: `.gmt` files converted to variant-level annotations with nearest-gene weighting; see [GMT Format](../file_formats.md#gmt-format-gmt).

## Output Files

Default output:

- `output_prefix.tall.csv`: heritability, enrichment, and coefficient estimates
- `output_prefix.convergence.csv`: optimization diagnostics

With `--alt-output`:

- `output_prefix.heritability.csv`
- `output_prefix.enrichment.csv`
- `output_prefix.parameters.csv`

Use `--name` to label runs when appending to alternate output files or score-test HDF5 outputs.

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--intercept` | `1.0` | LD score regression intercept |
| `--name` | `None` | Run label for outputs and score-test artifacts |
| `--metadata` | `data/ldgms/metadata.csv` | LDGM metadata CSV |
| `--score-test-filename` | `None` | HDF5 file for score-test precomputation |
| `--surrogates` | `None` | Precomputed surrogate-marker HDF5 |
| `--no-save` | `False` | Skip result-file writing or write logs only |

## Optimization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-iterations` | `50` | Maximum optimization iterations |
| `--convergence-tol` | `0.01` | Convergence tolerance |
| `--convergence-window` | `3` | Iterations used for convergence checks |
| `--num-jackknife-blocks` | `100` | Jackknife blocks for standard errors |
| `--xtrace-num-samples` | `100` | Samples for stochastic gradient estimation |
| `--reset-trust-region` | `False` | Reset trust-region size each iteration |
| `--initial-params` | `None` | Comma-separated initial coefficient values |

## Variant Matching And Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `--match-by-position` | `False` | Match variants by genomic position instead of RSID |
| `--maximum-missingness` | `0.1` | Maximum missing-variant fraction allowed |
| `--max-chisq-threshold` | `None` | Drop LD blocks above a chi-squared threshold |
| `--annotation-columns` | all | Restrict to specific annotation columns |
| `--binary-annotations-only` | `False` | Keep only 0/1-valued annotations |

## Gene Set Annotations

Use `--gene-annot-dir` to supply GMT files:

```bash
uv run graphld reml \
    /path/to/sumstats.sumstats \
    output_prefix \
    --gene-annot-dir /path/to/gmt/files/ \
    --gene-table data/genes.tsv
```

Related options:

| Option | Default | Description |
|--------|---------|-------------|
| `--gene-table` | `data/genes.tsv` | Gene coordinate table |
| `--nearest-weights` | `0.4,0.2,0.1,0.1,0.1,0.05,0.05` | Weights for nearest-gene mapping |

GMT rows are:

```text
gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
```

## Surrogate Markers

When GWAS variants are missing from the LDGM reference, graphREML can use surrogate markers in high LD. To avoid recomputing them for repeated analyses:

```bash
uv run graphld surrogates /path/to/sumstats.sumstats out.h5 --population EUR
uv run graphld reml /path/to/sumstats.sumstats output_prefix --annot-dir /path/to/annot --surrogates out.h5
```

## Parquet Files

For a multi-trait parquet input, select the trait to analyze when writing the
default saved output files:

```bash
uv run graphld reml sumstats.parquet output --name height
```

Default tall-output runs write one file pair per trait, such as
`output.height.tall.csv` and `output.height.convergence.csv`.
