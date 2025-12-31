# Command Line Interface

The GraphLD CLI provides commands for various LD-based analyses. Use `uv run graphld <command> -h` to see usage for a specific command.

For the enrichment score test CLI (`estest`), see [Enrichment Score Test](score_test.md).

## Available Commands

| Command | Description |
|---------|-------------|
| `reml` | Run graphREML heritability estimation |
| `blup` | Compute best linear unbiased predictor weights |
| `clump` | P-value thresholding and LD-based pruning |
| `simulate` | Simulate GWAS summary statistics |
| `surrogates` | Precompute surrogate markers |

## Heritability Estimation (reml)

Run graphREML for heritability partitioning:

```bash
uv run graphld reml \
    /path/to/sumstats/file.sumstats \
    output_files_prefix \
    --annot-dir /directory/containing/annotation/files/
```

### Input Formats

- **Summary statistics**: VCF (`.vcf`), LDSC (`.sumstats`), or Parquet (`.parquet`) format
- **Variant annotations**: Per-chromosome [LDSC `.annot` files](https://github.com/bulik/ldsc/wiki/LD-File-Formats#annot), including `thin-annot` format
- **BED files**: UCSC `.bed` files (not stratified per-chromosome) for binary annotations based on GRCh38 coordinates
- **Gene annotations**: GMT files (`.gmt`) containing gene sets, converted to variant-level using nearest-gene weighting

### Output Files

**Default output:**

- `output_prefix.tall.csv`: Heritability, enrichment, and coefficient estimates for each annotation
- `output_prefix.convergence.csv`: Information about the optimization process

**With `--alt-output` flag** (useful for storing results from multiple traits or runs in a single file):

- `output_prefix.heritability.csv`: Heritability estimates (one row per run)
- `output_prefix.enrichment.csv`: Enrichment estimates (one row per run)
- `output_prefix.parameters.csv`: Coefficient estimates (one row per run)

Use `--name` to label each run when appending to these files.

### Options Reference

```
uv run graphld reml --help
```

**Required (one of):**

| Option | Description |
|--------|-------------|
| `sumstats` | Path to summary statistics file (.vcf, .sumstats, or .parquet) |
| `-a, --annot-dir` | Path to annotation directory containing .annot and/or .bed files |
| `-g, --gene-annot-dir` | Path to directory containing .gmt gene set files |

**Common options:**

| Option | Default | Description |
|--------|---------|-------------|
| `out` | None | Output file path prefix |
| `--intercept` | 1.0 | LD score regression intercept (recommended to estimate with LDSC first) |
| `-n, --num-samples` | auto | Sample size (auto-detected from sumstats if available) |
| `-p, --population` | EUR | Population for LDGM selection |
| `-c, --chromosome` | all | Restrict analysis to specific chromosome |
| `--name` | None | Label for this run (used in alt-output files and score test HDF5) |
| `-v, --verbose` | False | Print detailed progress |
| `-q, --quiet` | False | Suppress output except errors |

**Optimization options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--num-iterations` | 50 | Maximum optimization iterations |
| `--convergence-tol` | 0.01 | Convergence tolerance |
| `--convergence-window` | 3 | Iterations to consider for convergence |
| `--num-jackknife-blocks` | 100 | Number of jackknife blocks for standard errors |
| `--xtrace-num-samples` | 100 | Samples for stochastic gradient estimation |
| `--reset-trust-region` | False | Reset trust region at each iteration |

**Processing options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--num-processes` | auto | Number of parallel processes |
| `--run-in-serial` | False | Disable parallelization |
| `--match-by-position` | False | Match variants by position instead of RSID |
| `--maximum-missingness` | 0.1 | Maximum fraction of missing samples allowed |
| `--max-chisq-threshold` | None | Exclude blocks with chi-squared above threshold |

**Annotation options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--annotation-columns` | all | Specific annotation columns to use |
| `--binary-annotations-only` | False | Only include 0/1 valued annotations |

**Gene annotation options** (used with `-g, --gene-annot-dir`):

| Option | Default | Description |
|--------|---------|-------------|
| `--gene-table` | data/genes.tsv | Path to gene table TSV file |
| `--nearest-weights` | 0.4,0.2,0.1,0.1,0.1,0.05,0.05 | Weights for k-nearest genes |

**Advanced options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--surrogates` | None | Path to precomputed surrogate markers HDF5 |
| `--initial-params` | None | Initial parameter values (comma-separated) |
| `--metadata` | data/ldgms/metadata.csv | Path to LDGM metadata file |
| `--score-test-filename` | None | Output HDF5 for score test precomputation |
| `--alt-output` | False | Write separate files for heritability/enrichment/parameters |
| `--no-save` | False | Do not save results |

### Gene Set Annotations

GraphREML can use gene-level annotations from GMT files (Gene Matrix Transposed format). Gene sets are converted to variant-level annotations using a nearest-gene weighting scheme.

```bash
uv run graphld reml \
    /path/to/sumstats.sumstats \
    output_prefix \
    --gene-annot-dir /path/to/gmt/files/ \
    --gene-table data/genes.tsv
```

**GMT file format:**
```
gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
```

Each gene set becomes a variant-level annotation where each variant receives a weighted score based on its proximity to genes in the set. The `--nearest-weights` option controls how the k-nearest genes contribute to each variant's score.

The gene table TSV must contain columns: `gene_id`, `gene_name`, `start`, `end`, `CHR`.

### Surrogate Markers

If variants are missing from your GWAS summary statistics, graphREML automatically assigns surrogate markers in high LD. This is common when using HapMap3 SNPs only (~1.1M SNPs).

To speed up repeated analyses, precompute surrogates:

```bash
uv run graphld surrogates /path/to/sumstats.sumstats
```

Then pass the cached file with `--surrogates`. Surrogates don't need to match exactly and can be reused across similar sumstats files.

Precomputed surrogates for common reference panels are available for download from [Zenodo](https://zenodo.org/records/8157131) (TODO: add direct link).

### Parquet Multi-Trait Files

When using parquet files with multiple traits:

```bash
# Process specific traits
uv run graphld reml sumstats.parquet output --name height,bmi ...

# Process all traits (omit --name)
uv run graphld reml sumstats.parquet output ...
```

## BLUP Weights (blup)

Compute best linear unbiased predictor effect size estimates:

```bash
uv run graphld blup \
    /path/to/sumstats.sumstats \
    output_prefix \
    --heritability 0.1
```

## LD Clumping (clump)

Perform p-value thresholding and LD-based pruning:

```bash
uv run graphld clump \
    /path/to/sumstats.sumstats \
    output_prefix \
    --rsq-threshold 0.1 \
    --chisq-threshold 30.0
```

## Simulation (simulate)

Simulate GWAS summary statistics:

```bash
uv run graphld simulate \
    output_prefix \
    --sample-size 10000 \
    --heritability 0.5
```
