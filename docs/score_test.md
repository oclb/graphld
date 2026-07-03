# Enrichment Score Test

The enrichment score test is a fast method to test genomic or gene annotations for heritability enrichment conditional upon a null model.

## Overview

The test produces Z scores:

- **Positive score**: Heritability enrichment
- **Negative score**: Heritability depletion

Enrichments are conditional upon the null model, similar to the `tau` parameter in S-LDSC. The test does not produce point estimates (for that, run graphREML).

## Requirements

You need a file containing precomputed derivatives for each trait being tested. This can be:

- Downloaded from Zenodo via the Makefile (`make download_scores`)
- Created by running graphREML with the `--score-test-filename` flag

## Supported Annotation Formats

| Format | Description |
|--------|-------------|
| [LDSC `.annot`](file_formats.md#ldsc-format-annot) | Variant annotations |
| [UCSC `.bed`](file_formats.md#bed-format-bed) | Genomic regions |
| [GMT](file_formats.md#gmt-format-gmt) | Gene sets (symbols or IDs) |

## Basic Usage

### View Available Traits

```bash
uv run estest show path/to/scores.h5
```

### Test Variant Annotations

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir path/to/annot_dir/
```

### Test Genomic Regions

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir path/to/bed_dir/
```

### Test Gene Annotations

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --gene-annot-dir path/to/gmt/files/
```

### Options Reference

```
uv run estest test --help
```

| Option | Description |
|--------|-------------|
| `-a, --variant-annot-dir` | Directory containing `.annot` and/or `.bed` files |
| `-g, --gene-annot-dir` | Directory containing .gmt files |
| `--random-genes` | Comma-separated probabilities for random gene annotations |
| `--random-variants` | Comma-separated probabilities for random variant annotations |
| `--gene-table` | Path to gene table TSV (required for gene-level options) |
| `--nearest-weights` | Comma-separated weights for k-nearest genes |
| `--annotations` | Specific annotation names to test |
| `-n, --name` | Specific trait to process (default: all traits) |
| `-v, --verbose` | Enable verbose output |
| `--seed` | Random seed for reproducibility |

## Random Annotations

Test random annotations to verify the null distribution:

### Random Variant Annotations

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --random-variants 0.1,0.2,0.3
```

Creates random annotations with 10%, 20%, and 30% of variants.

### Random Gene Annotations

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --random-genes 0.1,0.2,0.3
```

Creates random annotations with 10%, 20%, and 30% of genes.

### Perturb Existing Annotations

```bash
uv run estest \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --variant-annot-dir path/to/annot_dir/ \
    --perturb-annot 0.5  # 50% of annotation values sampled randomly
```

## Gene Set Testing

Gene sets can be tested for heritability enrichment under the Abstract Mediation Model (AMM; Weiner et al. 2022 AJHG). This tests whether variants in proximity to genes in the gene set are enriched for heritability.

### Basic Approach

Supply a GMT file to `--gene-annot-dir`.

### Faster Approach

Convert variant-level to gene-level score statistics first:

```bash
uv run estest convert path/to/scores.h5 path/to/gene_scores.h5
```

This requires a gene positions file (provided in `data/genes.tsv` after running the Makefile).

Then run the test:

```bash
uv run estest \
    path/to/gene_scores.h5 \
    path/to/output/file/prefix \
    --gene-annot-dir path/to/gmt/files/
```

Results are nearly identical to the variant-level test but much faster.

## Meta-Analysis Across Traits

Test whether an annotation is enriched across multiple traits:

### Add Meta-Analysis

```bash
# Add all traits
uv run estest add-meta path/to/scores.h5 all_traits '*'

# Add specific traits
uv run estest add-meta path/to/scores.h5 body_traits height bmi
```

Then run the score test as normal. The meta-analysis appears as a column in the output.

The meta-analysis uses precision-weighted linear combination of score statistics with jackknife standard errors. Non-independence across traits causes power loss but not false positives.

## Manipulating HDF5 Files

The `estest` command provides utilities for managing traits and meta-analyses in HDF5 files.

### Show Contents

Display all traits and meta-analyses in an HDF5 file:

```bash
uv run estest show path/to/scores.h5
```

### Rename Traits or Meta-Analyses

Rename a trait or meta-analysis (auto-detects which):

```bash
uv run estest mv path/to/scores.h5 old_name new_name
```

### Remove Traits or Meta-Analyses

Remove one or more traits or meta-analyses:

```bash
# Remove a single item
uv run estest rm path/to/scores.h5 trait_name

# Remove multiple items
uv run estest rm path/to/scores.h5 bmi height cancer

# Use wildcards
uv run estest rm path/to/scores.h5 '*_EAS'

# Force removal without confirmation
uv run estest rm path/to/scores.h5 'BMI*' -f
```

The command auto-detects whether names are traits or meta-analyses and supports wildcards (`*`).

## Creating Derivatives For A New Trait

First, run graphREML on the trait and ask it to write score-test derivatives:

```bash
uv run graphld reml \
    path/to/trait.sumstats \
    path/to/reml/output_prefix \
    --annot-dir path/to/null_model_annotations/ \
    --score-test-filename path/to/scores.h5 \
    --name trait_name \
    --population EUR
```

The annotations supplied to graphREML define the null model for the later score test. For example, use baseline annotations if you want to test for a conditional enrichment under that baseline model. If you have an annotation of particular interest, and you include it in the graphREML model run, then graphREML reports its enrichment and conditional enrichment estimates directly. A later score test of the exact same annotation is not useful because the annotation has already been projected out of the score.

Then confirm that the trait was written:

```bash
uv run estest show path/to/scores.h5
```

Finally, run `estest` on the new derivative file:

```bash
uv run estest test \
    path/to/scores.h5 \
    path/to/output/file/prefix \
    --gene-annot-dir path/to/gmt/files/ \
    --gene-table data/genes.tsv
```

Use `--variant-annot-dir` instead of `--gene-annot-dir` to test variant annotations or BED regions.

Notes:

- Creating derivatives requires full graphREML setup, including downloading LDGMs. Running `estest` on an existing derivative file does not.
- Multiple traits can be added to the same score-statistics file. The first trait added creates the file and defines its variant rows and jackknife assignments. Append additional traits only when they use the same score-test row set and row order, which means using the same LDGMs, population, chromosomes, matching settings, filters, and null-model annotations.
- `--name` becomes the trait name in the HDF5 file.
- Combining `--score-test-filename` with `--match-by-position` will raise an error.
