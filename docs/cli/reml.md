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

In the convergence CSV, `num_iterations` counts optimizer iterations. The
`accepted_step` history starts at zero for initialization and advances only when
a step is accepted.

With `--alt-output`:

- `output_prefix.heritability.csv`
- `output_prefix.enrichment.csv`
- `output_prefix.parameters.csv`

Use `--name` to label runs when appending to alternate output files or score-test HDF5 outputs.

To create score-test derivatives for a new trait, pass `--score-test-filename`.
See [Creating Derivatives For A New Trait](../score_test.md#creating-derivatives-for-a-new-trait)
for the full workflow, including the UKBB population setting used by the
downloaded European score file.

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
| `--optimizer` | `ai` | Average-information optimization with line search |
| `--num-iterations` | `100` | Maximum optimization iterations |
| `--convergence-tol` | `0.001` | Precise score and signed objective tolerance |
| `--convergence-window` | `3` | Compatibility argument |
| `--num-jackknife-blocks` | `100` | Jackknife blocks for standard errors |
| `--xtrace-num-samples` | `100` | Samples for stochastic gradient estimation |
| `--reset-trust-region` | `False` | Compatibility argument |
| `--initial-params` | `None` | Comma-separated initial coefficient values |
| `--link-function` | `softplus` | Variance link: `softplus` or `exponential` |
| `--information-penalty`, `--firth` | `0.0` | Optional information-penalty weight; `1.0` when enabled without a value |
| `--penalty-trial-strategy` | `exact` | Trial evaluation: `exact`, or experimental `linear_screen` / `likelihood_screen` |

graphREML uses average-information optimization with precise convergence checks.
The default tolerance of `0.001` bounds an information-scaled score criterion and
actual objective improvements in four signed direction checks. Earlier releases
used recent likelihood changes; `--convergence-window` and `--reset-trust-region`
are retained for compatibility and have no effect.

Inspect `converged`, `termination_reason`, and `uncertainty_status` in the
convergence CSV before interpreting estimates. Convergence establishes the local
checks, without guaranteeing a global optimum. Uncertainty at unresolved endpoints
is provisional; singular information or invalid delete calculations make it
unavailable.

To enable the optional information penalty:

```bash
uv run graphld reml summary.sumstats output --annot-dir annotations --firth
```

This adds one half the log determinant of global average information to the
likelihood. It requires positive-definite information at initialization. Use
`--information-penalty 0.5` for a different nonnegative weight. Formal bias
reduction and uncertainty calibration for penalized estimates have not been
established. See the [methods and evaluation write-up](../methods/graphreml_optimization/README.md)
for algorithm details, screening strategies, and numerical comparisons.

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
